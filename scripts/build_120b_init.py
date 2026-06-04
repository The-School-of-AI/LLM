"""
Build a proper 120B init checkpoint by:
  1. Instantiating the 120B model object (CPU)
  2. Loading the 9B source state_dict
  3. For every model.named_parameters() and named_buffers(), populating the
     value with the appropriate source — drop-upcycled for expert weights,
     direct copy for everything else
  4. Saving model.state_dict() — guaranteed to have the EXACT key set the
     model later expects, so load_state_dict(strict=True) will work

Why this approach: previous init scripts wrote keys in the 9B source path format
(`layers.X.mlp_block...`) but the 120B model expects different paths
(reversible-stack wrapping creates `stack.blocks.X...` AND `layers.X...` as
separate parameter sets). Loading source-format keys silently dropped weights into the
"layers.X" set while the real training params at `stack.blocks.X` stayed at
randn-fallback. By saving model.state_dict() we get all paths the model has.

Drop-Upcycling recipe (Nakamura et al., arXiv 2502.19261):
  For each target expert (independently):
    - Pick a random subset S of intermediate-dim indices, |S| = r * d_hidden
    - Replace those positions with N(μ, σ²) drawn from source's stats at S
    - Other positions stay as direct copies of the source

Indexing: clone-major (idx = c*20 + e) where c ∈ 0..22, e ∈ 0..19.
Same mask used for W_gate, W_up, W_down within an expert.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

N_CLONES = 23
N_SOURCE = 20
N_TARGET = N_CLONES * N_SOURCE  # 460


def is_main_backbone_path(p: str) -> bool:
    """Path is on main reversible backbone (not MTP block)."""
    return (not p.startswith("mtp_block")) and (
        p.startswith("layers.") or p.startswith("stack.")
    )


def is_expert_W_path(p: str) -> bool:
    return is_main_backbone_path(p) and any(
        p.endswith(s) for s in (".W_gate", ".W_up", ".W_down")
    )


def is_router_W_path(p: str) -> bool:
    return is_main_backbone_path(p) and p.endswith(".gate.gate.weight")


def is_logit_bias_path(p: str) -> bool:
    return is_main_backbone_path(p) and p.endswith(".gate.logit_bias")


def candidate_source_paths(model_path: str):
    """Yield source-state_dict paths to try for a given model path.

    The source MoE checkpoint uses `layers.N.mlp_block.sublayer....` paths. The 120B model
    sometimes wraps these in `stack.blocks.N...`, `stack.mid_layers.N...`,
    etc. We try a small list of prefix substitutions.
    """
    yield model_path
    # Common reversible-stack rewrites:
    rewrites = [
        ("stack.blocks.", "layers."),
        ("stack.mid_layers.", "layers."),
        ("reversible_stack.blocks.", "layers."),
        ("reversible_stack.mid_layers.", "layers."),
        # Walk inside .block.block.sublayer or .wrapper.layer.block etc.
        # but those are deep paths within a layer, not prefix-to-prefix maps.
    ]
    for src_prefix, dst_prefix in rewrites:
        if model_path.startswith(src_prefix):
            cand = dst_prefix + model_path[len(src_prefix) :]
            if cand != model_path:
                yield cand
    # Also handle inner "block.block" / "block" wrappers in mid_layers paths
    # e.g. layers.0.mlp_block.sublayer.moe.W_gate vs
    #      stack.mid_layers.0.block.block.mlp_block.sublayer.moe.W_gate
    inner_rewrites = [
        (".block.block.", "."),
        (".block.", "."),
        (".wrapper.layer.block.", "."),
        (".wrapper.layer.", "."),
    ]
    for innersub, replace in inner_rewrites:
        if innersub in model_path:
            cand = model_path.replace(innersub, replace)
            if cand != model_path:
                yield cand
    # Combined (prefix + inner) — try each pair
    for src_prefix, dst_prefix in rewrites:
        if model_path.startswith(src_prefix):
            inner = model_path[len(src_prefix) :]
            for innersub, replace in inner_rewrites:
                if innersub in inner:
                    yield dst_prefix + inner.replace(innersub, replace)


def find_source_tensor(model_path: str, src_sd: dict, expected_shape=None):
    """Try candidates; return (matched_source_path, src_tensor) or (None, None)."""
    for cand in candidate_source_paths(model_path):
        if cand in src_sd:
            return cand, src_sd[cand]
    return None, None


def drop_upcycle_one_expert_set(
    src_W_gate,
    src_W_up,
    src_W_down,
    e: int,
    c: int,
    gen,
    ratio: float = 0.5,
):
    """Build target-expert tensors for clone c of source-expert e."""
    src_g = src_W_gate.float()
    src_u = src_W_up.float()
    src_d = src_W_down.float()
    d_hidden = src_g.shape[-1]
    n_reinit = int(round(ratio * d_hidden))
    perm = torch.randperm(d_hidden, generator=gen)
    S = perm[:n_reinit].sort().values
    sel_g = src_g[:, S]
    mu_g, sigma_g = sel_g.mean().item(), sel_g.std().item()
    new_g = src_g.clone()
    new_g[:, S] = torch.empty_like(sel_g).normal_(mean=mu_g, std=sigma_g, generator=gen)
    sel_u = src_u[:, S]
    mu_u, sigma_u = sel_u.mean().item(), sel_u.std().item()
    new_u = src_u.clone()
    new_u[:, S] = torch.empty_like(sel_u).normal_(mean=mu_u, std=sigma_u, generator=gen)
    sel_d = src_d[S, :]
    mu_d, sigma_d = sel_d.mean().item(), sel_d.std().item()
    new_d = src_d.clone()
    new_d[S, :] = torch.empty_like(sel_d).normal_(mean=mu_d, std=sigma_d, generator=gen)
    return new_g, new_u, new_d


def build_460_expert_tensor(
    src_w20: torch.Tensor,  # [20, ...]
    is_W_down: bool,
    src_other_w20: dict,  # {'W_up': [20,...], 'W_down': [20,...]} so masks are joint per-clone
    gen,
    ratio: float,
):
    """Drop-upcycle a single 20-expert source into a 460-expert tensor.

    Joint masking across W_gate/W_up/W_down per (clone, source-expert) pair
    requires us to do all three simultaneously. This helper focuses on ONE
    of the three; the caller must pre-compute masks and reuse them.
    """
    raise NotImplementedError("Use build_layer_experts to do all three jointly.")


def build_layer_experts(src_W_gate, src_W_up, src_W_down, gen, ratio=0.5):
    """Build [460, ...] tensors for all three weight matrices of one layer."""
    assert src_W_gate.shape[0] == N_SOURCE
    d_model = src_W_gate.shape[1]
    d_hidden = src_W_gate.shape[2]
    out_g = torch.empty(N_TARGET, d_model, d_hidden, dtype=src_W_gate.dtype)
    out_u = torch.empty(N_TARGET, d_model, d_hidden, dtype=src_W_up.dtype)
    out_d = torch.empty(N_TARGET, d_hidden, d_model, dtype=src_W_down.dtype)
    for c in range(N_CLONES):
        for e in range(N_SOURCE):
            idx = c * N_SOURCE + e
            ng, nu, nd = drop_upcycle_one_expert_set(
                src_W_gate[e],
                src_W_up[e],
                src_W_down[e],
                e=e,
                c=c,
                gen=gen,
                ratio=ratio,
            )
            out_g[idx] = ng.to(out_g.dtype)
            out_u[idx] = nu.to(out_u.dtype)
            out_d[idx] = nd.to(out_d.dtype)
    return out_g, out_u, out_d


def tile_router(src_gate: torch.Tensor, sigma_relative: float, gen):
    """[20, d_model] -> [460, d_model] by tiling + small per-element noise."""
    assert src_gate.shape[0] == N_SOURCE
    tiled = src_gate.float().repeat(N_CLONES, 1)
    if sigma_relative > 0:
        per_tensor_std = src_gate.float().std().item()
        tiled = tiled + torch.empty_like(tiled).normal_(
            0, sigma_relative * per_tensor_std, generator=gen
        )
    return tiled.to(src_gate.dtype)


def populate_model_param(
    model_path: str,
    model_param: torch.Tensor,
    src_sd: dict,
    layer_expert_cache: dict,  # cache of drop-upcycled per-layer expert tensors
    gen,
    ratio: float,
    router_sigma: float,
    counts: dict,
):
    """Set `model_param.data` in-place from the appropriate source.

    Returns True if populated, False if no source found (leave at random init).
    """
    src_path, src_tensor = find_source_tensor(model_path, src_sd)

    # MTP block: copy unchanged (stays at 20 experts)
    if model_path.startswith("mtp_block"):
        if src_path is None:
            counts["mtp_no_src"] += 1
            return False
        if src_tensor.shape != model_param.shape:
            counts["mtp_shape_mismatch"] += 1
            print(
                f"  MTP shape mismatch {model_path}: model {tuple(model_param.shape)} src {tuple(src_tensor.shape)}"
            )
            return False
        model_param.data.copy_(src_tensor.to(model_param.dtype))
        counts["mtp_copied"] += 1
        return True

    # Backbone expert weights (W_gate / W_up / W_down): drop-upcycle
    if is_expert_W_path(model_path):
        if src_path is None:
            counts["expert_no_src"] += 1
            return False
        # Need all three of W_gate/W_up/W_down to do joint drop-upcycle.
        # Use a per-layer cache so we compute once and serve three calls.
        # The "layer key" is the source path with the trailing .W_X stripped.
        suffix = next(
            s for s in (".W_gate", ".W_up", ".W_down") if model_path.endswith(s)
        )
        layer_key = src_path[: -len(suffix)]
        if layer_key not in layer_expert_cache:
            wg = src_sd[layer_key + ".W_gate"]
            wu = src_sd[layer_key + ".W_up"]
            wd = src_sd[layer_key + ".W_down"]
            ug, uu, ud = build_layer_experts(wg, wu, wd, gen, ratio)
            layer_expert_cache[layer_key] = {"W_gate": ug, "W_up": uu, "W_down": ud}
        new_t = layer_expert_cache[layer_key][suffix.lstrip(".")]
        if new_t.shape != model_param.shape:
            counts["expert_shape_mismatch"] += 1
            print(
                f"  Expert shape mismatch {model_path}: model {tuple(model_param.shape)} new {tuple(new_t.shape)}"
            )
            return False
        model_param.data.copy_(new_t.to(model_param.dtype))
        counts["expert_drop_upcycle"] += 1
        return True

    # Backbone router weight: tile + small Gaussian
    if is_router_W_path(model_path):
        if src_path is None:
            counts["router_no_src"] += 1
            return False
        new_t = tile_router(src_sd[src_path], router_sigma, gen)
        if new_t.shape != model_param.shape:
            counts["router_shape_mismatch"] += 1
            print(
                f"  Router shape mismatch {model_path}: model {tuple(model_param.shape)} new {tuple(new_t.shape)}"
            )
            return False
        model_param.data.copy_(new_t.to(model_param.dtype))
        counts["router_tiled"] += 1
        return True

    # Backbone logit_bias: 460-dim zeros
    if is_logit_bias_path(model_path):
        if model_param.numel() == N_TARGET:
            model_param.data.zero_()
            counts["logit_bias_zeroed"] += 1
            return True
        # MTP-style 20-dim: copy from source
        if src_path is not None and src_sd[src_path].shape == model_param.shape:
            model_param.data.copy_(src_sd[src_path].to(model_param.dtype))
            counts["logit_bias_copied"] += 1
            return True
        counts["logit_bias_no_src"] += 1
        return False

    # Everything else: direct copy if shape matches
    if src_path is not None and src_sd[src_path].shape == model_param.shape:
        model_param.data.copy_(src_sd[src_path].to(model_param.dtype))
        counts["copied_other"] += 1
        return True
    if src_path is not None:
        counts["other_shape_mismatch"] += 1
        if counts["other_shape_mismatch"] <= 5:
            print(
                f"  Other shape mismatch {model_path}: model {tuple(model_param.shape)} src {tuple(src_sd[src_path].shape)}"
            )
        return False
    counts["other_no_src"] += 1
    if counts["other_no_src"] <= 10:
        print(f"  No source for {model_path} (model shape {tuple(model_param.shape)})")
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="9B consolidated .pt")
    p.add_argument(
        "--dst", required=True, help="Output 120B init .pt (model.state_dict())"
    )
    p.add_argument(
        "--config",
        default="configs/train_120b_tqp.yaml",
        help="Path to train_120b_tqp.yaml",
    )
    p.add_argument("--ratio", type=float, default=0.5)
    p.add_argument("--router_sigma", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    print("=== build_120b_init ===")
    print(f"src     : {args.src}")
    print(f"dst     : {args.dst}")
    print(f"r       : {args.ratio}")
    print(f"router_σ: {args.router_sigma}")
    print(f"seed    : {args.seed}")

    t0 = time.time()
    print("\nLoading 9B source state_dict...")
    src_sd = torch.load(args.src, map_location="cpu", weights_only=False)
    if (
        isinstance(src_sd, dict)
        and "state_dict" in src_sd
        and isinstance(src_sd["state_dict"], dict)
    ):
        src_sd = src_sd["state_dict"]
    elif (
        isinstance(src_sd, dict)
        and "module" in src_sd
        and isinstance(src_sd["module"], dict)
    ):
        src_sd = src_sd["module"]
    print(f"  {len(src_sd)} src keys, loaded in {time.time()-t0:.1f}s")

    print("\nBuilding 120B model object on CPU (this can take several minutes)...")
    t1 = time.time()
    # Mimic main.py's model construction path. The simplest robust way is to
    # call into the same build helpers main.py uses.
    import yaml

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    # Build the same model as main.py: Model120B with Kronecker embeddings.
    from lightninglm.models.recurrence_model_120b_moe import (
        KroneckerConfig,
        KroneckerEmbeddings,
        Model120B,
        ModelConfig,
    )

    mc = ModelConfig()
    mc.vocab_size = 131072
    mc.num_real_experts = int(cfg["model"].get("num_real_experts", 460))
    mc.num_null_experts = int(cfg["model"].get("num_null_experts", 0))
    mc.total_expert_slots = mc.num_real_experts + mc.num_null_experts
    print(
        f"  num_real_experts={mc.num_real_experts}, "
        f"num_null_experts={mc.num_null_experts}, "
        f"top_k={mc.top_k}"
    )

    # Tokenizer: try multiple paths
    _tok_dir_cfg = cfg["data"].get("tokenizer_dir", "tokenizer")
    _tok_path = Path(_tok_dir_cfg).expanduser()
    _tok_candidates = [_tok_path]
    if not _tok_path.is_absolute():
        _tok_candidates.extend(
            [
                (Path(args.config).resolve().parent / _tok_path).resolve(),
                (repo_root / _tok_path).resolve(),
            ]
        )
    tokenizer = None
    for _td in _tok_candidates:
        try:
            from lightninglm.data.data import get_tokenizer

            tokenizer = get_tokenizer(str(_td))
            print(f"  Loaded tokenizer from: {_td}")
            break
        except Exception:
            continue
    if tokenizer is None:
        raise SystemExit("Could not load tokenizer from any candidate path")

    # Build bpe_vocab list and pf_codec (Kronecker) per main.py's pattern
    vocab_size = len(tokenizer)
    mc.vocab_size = vocab_size
    print(f"  vocab_size: {vocab_size}")
    print(f"  Building bpe_vocab list (decoding {vocab_size} tokens)...")
    bpe_vocab = []
    for i in range(vocab_size):
        try:
            token = tokenizer.decode([i])
            bpe_vocab.append(token if token else f"<unk_{i}>")
        except Exception:
            bpe_vocab.append(f"<unk_{i}>")

    print("  Building Kronecker pf_codec (CHAR_DIM=256 POS_DIM=32 D=8192)...")
    pf_config = KroneckerConfig(
        CHAR_DIM=256,
        POS_DIM=32,
        D=8192,
        length_normalize=True,
        truncate_long_words=True,
    )
    pf_codec = KroneckerEmbeddings(pf_config)

    print("  Instantiating Model120B (this is the long part on CPU, ~5-10 min)...")
    model = Model120B(
        config=mc,
        embedding_type="kronecker",
        bpe_vocab=bpe_vocab,
        pf_codec=pf_codec,
    )
    model = model.to(dtype=torch.bfloat16)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model built in {time.time()-t1:.1f}s, {n_params/1e9:.2f}B params")

    print("\nPopulating model parameters from source...")
    t2 = time.time()
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    layer_expert_cache = {}
    counts = {
        "expert_drop_upcycle": 0,
        "expert_no_src": 0,
        "expert_shape_mismatch": 0,
        "router_tiled": 0,
        "router_no_src": 0,
        "router_shape_mismatch": 0,
        "logit_bias_zeroed": 0,
        "logit_bias_copied": 0,
        "logit_bias_no_src": 0,
        "mtp_copied": 0,
        "mtp_no_src": 0,
        "mtp_shape_mismatch": 0,
        "copied_other": 0,
        "other_no_src": 0,
        "other_shape_mismatch": 0,
    }
    n_total_param = 0
    for name, p in model.named_parameters():
        n_total_param += 1
        populate_model_param(
            name,
            p,
            src_sd,
            layer_expert_cache,
            gen,
            args.ratio,
            args.router_sigma,
            counts,
        )
    print(f"  populated {n_total_param} parameters in {time.time()-t2:.1f}s")
    # Buffers (rotation matrices, codebooks, logit_bias, etc.)
    n_total_buf = 0
    for name, b in model.named_buffers():
        n_total_buf += 1
        # Most buffers aren't trainable and don't need population; only logit_bias-style
        # buffers are touched here. Skip rotation/codebook (they're recomputed at TQ wrap).
        if is_logit_bias_path(name):
            populate_model_param(
                name,
                b,
                src_sd,
                layer_expert_cache,
                gen,
                args.ratio,
                args.router_sigma,
                counts,
            )

    print("\nCounts:")
    for k, v in counts.items():
        print(f"  {k:30s}: {v}")
    print(f"  total params: {n_total_param}")
    print(f"  total buffers: {n_total_buf}")

    # Sanity: how many "no_src" or shape-mismatch?
    no_src_total = sum(counts[k] for k in counts if k.endswith("_no_src"))
    mismatch_total = sum(counts[k] for k in counts if "mismatch" in k)
    print(f"\nUnpopulated: {no_src_total}, Shape mismatches: {mismatch_total}")
    print(
        "(Unpopulated will stay at model's init values — typically TQP adapter params, "
        "rotation matrices, codebooks. Shape mismatches are bugs to investigate.)"
    )

    print(f"\nSaving model.state_dict() to {args.dst}...")
    t3 = time.time()
    torch.save(model.state_dict(), args.dst)
    print(f"  saved in {time.time()-t3:.1f}s")
    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
