from typing import Tuple, Dict
import gradio as gr


def dense_transformer_flops(
    n_layer: int,
    d_model: int,
    d_ff: int,
    d_attn: int,
    n_ctx: int,
    n_vocab: int,
    n_heads: int,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """
    Calculate FLOPs and parameter counts for a standard Dense Transformer.

    Args:
        n_layer: Number of transformer layers
        d_model: Model hidden dimension
        d_ff: Feed-forward network dimension
        d_attn: Attention head dimension (usually d_model // n_heads)
        n_ctx: Sequence length (context window)
        n_vocab: Vocabulary size
        n_heads: Number of attention heads

    Returns:
        flops_terms: Tuple of FLOPs for each component
        param_estimates: Tuple of parameter estimates for embeddings, layers, and output logits
    """
    
    # Embedding lookup is not a dense GEMM; ignore FLOPs to avoid overcount.
    embeddings_flops = 0.0

    # Attention FLOPs per layer
    attn_qkv_flops = 2 * n_ctx * 3 * d_model * (d_attn * n_heads)       # Q, K, V projections
    attn_logits_flops = 2 * n_ctx * n_ctx * (d_attn * n_heads)          # Attention score computation
    attn_softmax_flops = 3 * n_heads * n_ctx * n_ctx                    # Softmax operations
    attn_reduce_flops = 2 * n_ctx * n_ctx * (d_attn * n_heads)          # Weighted sum of values
    attn_proj_flops = 2 * n_ctx * (d_attn * n_heads) * d_model          # Output projection

    # Feed-forward FLOPs per layer (2 linear layers)
    ff_flops = 2 * n_ctx * (d_model * d_ff + d_ff * d_model)

    # Output logits FLOPs
    logits_flops = 2 * n_ctx * d_model * n_vocab

    # Parameter estimates (for embeddings, dense layers, and output projection)
    embedding_params = n_vocab * d_model
    layer_params = n_layer * (3 * d_model * d_model +  # QKV
                              d_model * d_model +      # Attention projection
                              2 * d_model * d_ff)      # FFN
    logits_params = d_model * n_vocab

    # Aggregate FLOPs across all layers
    flops_terms = (
        embeddings_flops,
        attn_qkv_flops * n_layer,
        attn_logits_flops * n_layer,
        attn_softmax_flops * n_layer,
        attn_reduce_flops * n_layer,
        attn_proj_flops * n_layer,
        ff_flops * n_layer,
        logits_flops
    )

    param_estimates = (
        embedding_params,
        layer_params,
        logits_params
    )

    return flops_terms, param_estimates


def moe_layer_flops(
    n_ctx: int,
    d_model: int,
    d_ff: int,
    num_experts: int,
    experts_per_token: int,
    load_balance_factor: float = 1.0,
    router_type: str = "top-k"
) -> Tuple[float, float, Dict[str, float]]:
    """
    Calculate FLOPs and active parameters for a single MoE (Mixture-of-Experts) layer.

    Args:
        n_ctx: Sequence length (context window)
        d_model: Model hidden dimension
        d_ff: Feed-forward network dimension per expert
        num_experts: Total number of experts in the layer
        experts_per_token: Number of experts activated per token (top-k)
        load_balance_factor: Overhead factor for load balancing (default 1.0)
        router_type: Type of router ("top-k", "switch", "expert-choice")

    Returns:
        total_flops: Total FLOPs for this MoE layer (including routing)
        active_params: Active parameters used per forward pass
        breakdown: Dictionary with detailed FLOPs breakdown
    """

    # Router FLOPs: Linear projection from hidden state to all experts
    router_flops = 2 * n_ctx * d_model * num_experts

    # Expert FLOPs (only for the active experts per token)
    # Each expert has two linear layers: d_model -> d_ff -> d_model
    expert_flops_per_token = experts_per_token * (
        2 * d_model * d_ff +   # First linear layer
        2 * d_ff * d_model     # Second linear layer
    )
    expert_flops_total = n_ctx * expert_flops_per_token

    # Apply load balancing overhead
    effective_expert_flops = expert_flops_total * load_balance_factor

    # Additional routing overhead depending on router type
    if router_type == "expert-choice":
        # Expert-choice routing requires extra selection operations
        routing_overhead = 2 * n_ctx * num_experts
    else:
        routing_overhead = 0

    # Total FLOPs for the layer
    total_flops = router_flops + effective_expert_flops + routing_overhead

    # Active parameters (only those used per forward pass)
    active_params_per_token = (
        experts_per_token * (d_model * d_ff + d_ff * d_model)
        + d_model * num_experts
    )
    active_params = active_params_per_token

    # Detailed FLOPs breakdown
    breakdown = {
        "router": router_flops,
        "experts": effective_expert_flops,
        "routing_overhead": routing_overhead,
        "total": total_flops
    }

    return total_flops, active_params, breakdown


from typing import Tuple, Dict

def moe_transformer_flops(
    n_layer: int,
    d_model: int,
    d_ff: int,
    d_attn: int,
    n_ctx: int,
    n_vocab: int,
    n_heads: int,
    num_experts: int,
    experts_per_token: int,
    moe_layers_start: int = 0,
    moe_layers_end: int = None,
    load_balance_factor: float = 1.0,
    router_type: str = "top-k",
    quantization_bits: int = 16,
) -> Tuple[Tuple[float, ...], Tuple[float, ...], Dict]:
    """
    Calculate FLOPs and parameter counts for a MoE Transformer.

    Args:
        n_layer: Number of transformer layers
        d_model: Model hidden dimension
        d_ff: Feed-forward network dimension
        d_attn: Attention head dimension
        n_ctx: Sequence length (context window)
        n_vocab: Vocabulary size
        n_heads: Number of attention heads
        num_experts: Total number of MoE experts
        experts_per_token: Number of experts activated per token (top-k)
        moe_layers_start: Layer index where MoE starts (0-indexed)
        moe_layers_end: Layer index where MoE ends (None = all remaining layers)
        load_balance_factor: Overhead factor for load balancing (1.0 = none)
        router_type: Type of router ("top-k", "switch", "expert-choice")
        quantization_bits: Bit precision (16=bf16, 8=int8, 4=MXFP4)

    Returns:
        flops_terms: Tuple of FLOPs for embeddings, attention, dense FFN, MoE, logits, total
        param_estimates: Tuple of total and active parameters
        breakdown: Detailed FLOPs and parameters dictionary
    """

    if moe_layers_end is None:
        moe_layers_end = n_layer

    # Embedding lookup is not a dense GEMM; ignore FLOPs to avoid overcount.
    embeddings_flops = 0.0

    # Output logits FLOPs
    logits_flops = 2 * n_ctx * d_model * n_vocab

    # Attention FLOPs per layer
    attn_qkv_flops = 2 * n_ctx * 3 * d_model * (d_attn * n_heads)      # Q, K, V projections
    attn_logits_flops = 2 * n_ctx * n_ctx * (d_attn * n_heads)         # Attention score computation
    attn_softmax_flops = 3 * n_heads * n_ctx * n_ctx                    # Softmax
    attn_reduce_flops = 2 * n_ctx * n_ctx * (d_attn * n_heads)          # Weighted sum of values
    attn_proj_flops = 2 * n_ctx * (d_attn * n_heads) * d_model          # Output projection
    total_attn_flops = n_layer * (attn_qkv_flops + attn_logits_flops +
                                  attn_softmax_flops + attn_reduce_flops +
                                  attn_proj_flops)

    # Dense FFN layers before/after MoE
    num_dense_layers = n_layer - (moe_layers_end - moe_layers_start)
    dense_ff_flops = num_dense_layers * 2 * n_ctx * (d_model * d_ff + d_ff * d_model)

    # MoE layers FLOPs and active parameters
    num_moe_layers = moe_layers_end - moe_layers_start
    moe_total_flops = 0
    moe_active_params = 0
    moe_breakdown = {}

    if num_moe_layers > 0:
        layer_flops, layer_params, layer_breakdown = moe_layer_flops(
            n_ctx, d_model, d_ff, num_experts, experts_per_token,
            load_balance_factor, router_type
        )
        moe_total_flops = num_moe_layers * layer_flops
        moe_active_params = layer_params
        moe_breakdown = {k: v * num_moe_layers for k, v in layer_breakdown.items()}

    # Quantization overhead factor
    quant_factor = 1.0
    if quantization_bits == 4:
        quant_factor = 1.05  # MXFP4 slight overhead
    elif quantization_bits == 8:
        quant_factor = 1.02  # INT8 slight overhead

    # Total FLOPs
    total_flops = embeddings_flops + total_attn_flops + dense_ff_flops + \
                  moe_total_flops * quant_factor + logits_flops

    # Total parameters
    total_params = (
        n_vocab * d_model +                 # Embeddings
        n_layer * (4 * d_model * d_model) + # Attention params (Q,K,V + proj)
        num_dense_layers * (2 * d_model * d_ff) + # Dense FFN
        num_moe_layers * (d_model * num_experts) + # MoE routers
        num_moe_layers * num_experts * (2 * d_model * d_ff) + # All MoE experts
        d_model * n_vocab                   # Output logits
    )

    # Active parameters (parameters actually used per forward pass)
    active_params = (
        n_vocab * d_model +                 # Embeddings
        n_layer * (4 * d_model * d_model) + # Attention params
        num_dense_layers * (2 * d_model * d_ff) + # Dense FFN
        num_moe_layers * (d_model * num_experts) + # MoE routers
        num_moe_layers * moe_active_params + # Only active experts
        d_model * n_vocab                   # Output logits
    )

    # Detailed breakdown dictionary
    breakdown = {
        "embeddings": embeddings_flops,
        "attention": total_attn_flops,
        "dense_ffn": dense_ff_flops,
        "moe_router": moe_breakdown.get("router", 0),
        "moe_experts": moe_breakdown.get("experts", 0),
        "moe_routing_overhead": moe_breakdown.get("routing_overhead", 0),
        "logits": logits_flops,
        "moe_total": moe_total_flops,
        "total": total_flops,
        "quantization_factor": quant_factor,
        "total_params": total_params,
        "active_params": active_params,
        "sparsity_ratio": active_params / total_params if total_params > 0 else 0
    }

    # Return detailed breakdown
    flops_terms = (
        embeddings_flops,
        total_attn_flops,
        dense_ff_flops,
        moe_total_flops,
        logits_flops,
        total_flops
    )
    param_estimates = (total_params, active_params)

    return flops_terms, param_estimates, breakdown


def calculator(
    model_type: str,
    n_layer: int,
    d_model: int,
    n_heads: int,
    n_vocab: int,
    ff_ratio: int,
    n_ctx: int,
    n_tokens: int,
    # MoE specific parameters
    num_experts: int,
    experts_per_token: int,
    moe_layers_start: int,
    moe_layers_end: int,
    load_balance_factor: float,
    router_type: str,
    quantization_bits: int,
    # Standard parameters
    incl_embed: bool,
    fwd_only: bool,
) -> Tuple:
    """
    Main calculator function supporting both Dense and MoE Transformers.

    Args:
        model_type: "Dense Transformer" or "MoE Transformer"
        n_layer: Number of transformer layers
        d_model: Model hidden dimension
        n_heads: Number of attention heads
        n_vocab: Vocabulary size
        ff_ratio: FFN expansion ratio (d_ff / d_model)
        n_ctx: Sequence length
        n_tokens: Total training tokens (optional)
        num_experts: Number of experts in MoE
        experts_per_token: Number of experts activated per token (top-k)
        moe_layers_start: Layer index where MoE starts
        moe_layers_end: Layer index where MoE ends
        load_balance_factor: Load balancing overhead for MoE
        router_type: Router type for MoE ("top-k", "switch", "expert-choice")
        quantization_bits: Quantization bit-width (16, 8, 4)
        incl_embed: Whether to include embeddings & logits in FLOPs
        fwd_only: Forward-only pass (if False, applies 3x for training)

    Returns:
        Tuple containing:
        - total_params
        - active_params
        - sparsity_ratio
        - efficiency_gain
        - flops_per_sequence
        - flops_per_token
        - n_tokens_flops
        - breakdown_text
    """
    
    # Derived dimensions
    d_attn = d_model // n_heads
    if d_model % n_heads != 0:
        raise gr.Error("d_model must be divisible by n_heads")
    
    d_ff = d_model * ff_ratio

    # Dense Transformer
    if model_type == "Dense Transformer":
        # Compute FLOPs and parameters
        flops_terms, params_tuple = dense_transformer_flops(
            n_layer, d_model, d_ff, d_attn, n_ctx, n_vocab, n_heads
        )

        # Unpack FLOPs and parameters
        embeddings_flop, attn_flop, attn_logits_flop, attn_softmax_flop, attn_reduce_flop, attn_proj_flop, ff_flop, logits_flop = flops_terms
        embedding_param, layer_param, logits_param = params_tuple

        # Include/exclude embeddings and logits
        if incl_embed:
            flops_per_sequence = sum(flops_terms)
            total_params = sum(params_tuple)
        else:
            flops_per_sequence = sum(flops_terms[1:-1])
            total_params = sum(params_tuple[1:])

        active_params = total_params
        sparsity_ratio = 1.0  # Dense: all parameters active

        # Detailed breakdown
        breakdown_text = f"""
**Dense Transformer Breakdown:**
- Total Parameters: {total_params:,.0f}
- Active Parameters: {active_params:,.0f}
- Sparsity Ratio: {sparsity_ratio:.2%}
- Embeddings FLOPs: {embeddings_flop:,.0f}
- Attention QKV FLOPs: {attn_flop:,.0f}
- Attention logits FLOPs: {attn_logits_flop:,.0f}
- Attention softmax FLOPs: {attn_softmax_flop:,.0f}
- Attention reduce FLOPs: {attn_reduce_flop:,.0f}
- Attention projection FLOPs: {attn_proj_flop:,.0f}
- FFN FLOPs: {ff_flop:,.0f}
- Output logits FLOPs: {logits_flop:,.0f}
"""

    # MoE Transformer
    else:
        if moe_layers_end == 0:
            moe_layers_end = n_layer

        # Compute MoE FLOPs and parameters
        flops_terms, params_tuple, breakdown = moe_transformer_flops(
            n_layer, d_model, d_ff, d_attn, n_ctx, n_vocab, n_heads,
            num_experts, experts_per_token, moe_layers_start, moe_layers_end,
            load_balance_factor, router_type, quantization_bits
        )

        total_params, active_params = params_tuple

        # Include/exclude embeddings & logits
        if incl_embed:
            flops_per_sequence = breakdown["total"]
        else:
            flops_per_sequence = breakdown["total"] - breakdown["embeddings"] - breakdown["logits"]

        sparsity_ratio = breakdown["sparsity_ratio"]

        # Detailed MoE breakdown
        breakdown_text = f"""
**MoE Transformer Breakdown:**
- Total Parameters: {total_params:,.0f}
- Active Parameters: {active_params:,.0f}
- Sparsity Ratio: {sparsity_ratio:.2%}
- Attention FLOPs: {breakdown['attention']:,.0f}
- Dense FFN FLOPs: {breakdown['dense_ffn']:,.0f}
- MoE Router FLOPs: {breakdown['moe_router']:,.0f}
- MoE Experts FLOPs: {breakdown['moe_experts']:,.0f}
- Quantization Overhead: {breakdown['quantization_factor']:.2f}x
"""

    # Compute per-token FLOPs
    flops_per_token = flops_per_sequence / n_ctx
    n_tokens_flops = flops_per_token * n_tokens if n_tokens > 0 else 0

    # Apply forward/backward multiplier
    if not fwd_only:
        flops_per_sequence *= 3
        flops_per_token *= 3
        n_tokens_flops *= 3

    # Parameter efficiency
    efficiency_gain = total_params / active_params if active_params > 0 else 1.0

    return (
        total_params,
        active_params,
        sparsity_ratio,
        efficiency_gain,
        flops_per_sequence,
        flops_per_token,
        n_tokens_flops,
        breakdown_text
    )


# Gradio Interface
with gr.Blocks(title="Dense & MoE Transformer FLOPs Calculator") as iface:
    gr.Markdown("""
    # Dense & Mixture-of-Experts (MoE) Transformer FLOPs Calculator
    
    This interactive calculator estimates **FLOPs and parameter usage** for both **Dense Transformers** and **Mixture-of-Experts (MoE) Transformers**, based on configurations of pre-trained models.
    
    Features include:
    - Dense and MoE model support
    - Top-k expert routing for MoE layers
    - Native quantization support (MXFP4, INT8, BF16)
    - Customizable MoE layer ranges
    - Load balancing overhead for expert activation
    
    This tool is inspired by [DeepMind's Chinchilla paper](https://arxiv.org/abs/2203.15556) and extended for sparse MoE architectures, providing detailed per-component FLOPs and parameter breakdowns.
    """)

    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Model Configuration")
            
            model_type = gr.Radio(
                ["Dense Transformer", "MoE Transformer"],
                label="Model Type",
                value="MoE Transformer"
            )
            
            gr.Markdown("#### Basic Architecture")
            n_layer = gr.Number(label="Number of layers", value=32)
            d_model = gr.Number(label="Model dimension (d_model)", value=4096)
            n_heads = gr.Number(label="Number of attention heads", value=32)
            n_vocab = gr.Number(label="Vocabulary size", value=128256)
            ff_ratio = gr.Number(value=4, label="FFN expansion ratio")
            
            gr.Markdown("#### MoE Configuration")
            num_experts = gr.Number(
                label="Number of experts per MoE layer",
                value=8,
                visible=True
            )
            experts_per_token = gr.Number(
                label="Experts activated per token (top-k)",
                value=2,
                visible=True
            )
            
            with gr.Row():
                moe_layers_start = gr.Number(
                    label="MoE start layer (0-indexed)",
                    value=0,
                    visible=True
                )
                moe_layers_end = gr.Number(
                    label="MoE end layer (0=all)",
                    value=0,
                    visible=True
                )
            
            load_balance_factor = gr.Slider(
                minimum=1.0,
                maximum=1.5,
                value=1.1,
                step=0.05,
                label="Load balance overhead factor",
                visible=True
            )
            
            router_type = gr.Radio(
                ["top-k", "switch", "expert-choice"],
                label="Router type",
                value="top-k",
                visible=True
            )
            
            quantization_bits = gr.Radio(
                [16, 8, 4],
                label="Quantization (bits)",
                value=4,
                visible=True
            )
            
            gr.Markdown("#### Data Configuration")
            n_ctx = gr.Number(label="Sequence length (context)", value=8192)
            n_tokens = gr.Number(
                value=0,
                label="Total training tokens (optional)",
            )
            
            gr.Markdown("#### Calculation Settings")
            incl_embed = gr.Checkbox(value=True, label="Include embeddings & logits")
            fwd_only = gr.Checkbox(
                value=False,
                label="Forward pass only (uncheck for training: 3x multiplier)"
            )
            
            btn = gr.Button(value="Calculate FLOPs", variant="primary", size="lg")
        
        with gr.Column():
            gr.Markdown("### Results")
            
            with gr.Row():
                total_params = gr.Number(label="Total Parameters")
                active_params = gr.Number(label="Active Parameters")
            
            with gr.Row():
                sparsity_ratio = gr.Number(label="Sparsity Ratio (active/total)")
                efficiency_gain = gr.Number(label="Parameter Efficiency Gain")
            
            gr.Markdown("#### FLOPs Metrics")
            flops_per_sequence = gr.Number(label="FLOPs per sequence")
            flops_per_token = gr.Number(label="FLOPs per token")
            n_tokens_flops = gr.Number(label="Total FLOPs for training")
            
            breakdown_text = gr.Markdown(label="Detailed Breakdown")
    
    # Event handlers
    def update_moe_visibility(model_type):
        visible = model_type == "MoE Transformer"
        return [
            gr.update(visible=visible),  # num_experts
            gr.update(visible=visible),  # experts_per_token
            gr.update(visible=visible),  # moe_layers_start
            gr.update(visible=visible),  # moe_layers_end
            gr.update(visible=visible),  # load_balance_factor
            gr.update(visible=visible),  # router_type
            gr.update(visible=visible),  # quantization_bits
        ]
    
    model_type.change(
        update_moe_visibility,
        inputs=[model_type],
        outputs=[
            num_experts,
            experts_per_token,
            moe_layers_start,
            moe_layers_end,
            load_balance_factor,
            router_type,
            quantization_bits,
        ]
    )
    
    btn.click(
        calculator,
        inputs=[
            model_type,
            n_layer,
            d_model,
            n_heads,
            n_vocab,
            ff_ratio,
            n_ctx,
            n_tokens,
            num_experts,
            experts_per_token,
            moe_layers_start,
            moe_layers_end,
            load_balance_factor,
            router_type,
            quantization_bits,
            incl_embed,
            fwd_only,
        ],
        outputs=[
            total_params,
            active_params,
            sparsity_ratio,
            efficiency_gain,
            flops_per_sequence,
            flops_per_token,
            n_tokens_flops,
            breakdown_text,
        ],
    )
    
    gr.Markdown("### Pre-configured Examples")
    
    
    with gr.Tab("Qwen 3 Series Models"):
      gr.Markdown("""
        **Qwen3-Dense-8B**: ~8B total params, ~8B active params  
        **Qwen3-MoE-30B**: ~30B total params, ~3.3B active params (sparse activation)
        """)

      gr.Examples(
          examples=[
              # ---------- Qwen 3 ----------
              [
                  "Dense Transformer", 36, 4096, 32, 151936, 12288/4096, 32768, 0,
                  1, 1, 0, 0, 1.0, "top-k", 16, True, False  # Qwen3-Dense-8B
              ],
              [
                  "MoE Transformer", 48, 2048, 32, 151936, 6144/2048, 32768, 0,
                  128, 8, 0, 48, 0.001, "top-k", 16, True, False  # Qwen3-MoE-30B
              ],
          ],
          inputs=[
              model_type, n_layer, d_model, n_heads, n_vocab, ff_ratio,
              n_ctx, n_tokens, num_experts, experts_per_token,
              moe_layers_start, moe_layers_end, load_balance_factor,
              router_type, quantization_bits, incl_embed, fwd_only,
          ],
          outputs=[
              total_params, active_params, sparsity_ratio, efficiency_gain,
              flops_per_sequence, flops_per_token, n_tokens_flops, breakdown_text
          ],
          fn=calculator,
          cache_examples=False,
      )


if __name__ == "__main__":
    iface.launch()
