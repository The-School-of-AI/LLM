
"""
spike_simulator.py

Simulates a rolling-window token sampler and checks domain spike constraints.

Inputs:
- A stream/list of samples, each with: tokens, domain_tag
- A target domain distribution (weights)

This tool helps catch "silent curriculum failures" where one domain dominates
in a rolling window even if global proportions look fine.

Usage:
  python spike_simulator.py --max-domain-share 0.25 --window-tokens 2000000

For real pipelines, feed it a sampled log from the dataloader.
"""

from __future__ import annotations

import argparse
import random
from collections import deque, defaultdict

def simulate_stream(domain_weights: dict, n_samples: int = 20000, token_mean: int = 800, token_jitter: int = 600, seed: int = 42):
    random.seed(seed)
    doms = list(domain_weights.keys())
    ws = [domain_weights[d] for d in doms]
    s = sum(ws)
    ws = [w/s for w in ws]

    for _ in range(n_samples):
        d = random.choices(doms, weights=ws, k=1)[0]
        tokens = max(50, int(random.gauss(token_mean, token_jitter)))
        yield {"domain": d, "tokens": tokens}

def check_spikes(stream, window_tokens: int, max_share: float):
    """
    Sliding window over token counts; ensures no domain exceeds max_share in any window.
    """
    window = deque()  # each entry: (domain, tokens)
    domain_tok = defaultdict(int)
    total = 0
    worst = (None, 0.0)

    for i, item in enumerate(stream):
        d, t = item["domain"], int(item["tokens"])
        window.append((d, t))
        domain_tok[d] += t
        total += t

        while total > window_tokens and window:
            od, ot = window.popleft()
            domain_tok[od] -= ot
            total -= ot
            if domain_tok[od] <= 0:
                del domain_tok[od]

        if total > 0:
            for dom, tok in domain_tok.items():
                share = tok / total
                if share > worst[1]:
                    worst = (dom, share)
                if share > max_share:
                    return False, {"step": i, "domain": dom, "share": share, "window_total_tokens": total, "worst_seen": worst}

    return True, {"worst_seen": worst, "window_tokens": window_tokens, "max_share": max_share}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-tokens", type=int, default=2_000_000)
    ap.add_argument("--max-domain-share", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--samples", type=int, default=20000)
    args = ap.parse_args()

    # Example domain weights; replace with your curriculum's domain targets
    domain_weights = {
        "general_web": 0.40,
        "wiki_books": 0.25,
        "code": 0.20,
        "papers_math": 0.10,
        "agentic": 0.05,
    }

    stream = simulate_stream(domain_weights, n_samples=args.samples, seed=args.seed)
    ok, info = check_spikes(stream, window_tokens=args.window_tokens, max_share=args.max_domain_share)

    if ok:
        print("OK: No domain spikes detected.")
        print("Worst seen:", info["worst_seen"])
    else:
        print("FAIL: Domain spike detected.")
        print(info)

if __name__ == "__main__":
    main()
