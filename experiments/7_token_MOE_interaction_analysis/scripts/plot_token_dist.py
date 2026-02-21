import argparse

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch

from moeint.expert_analysis import ModalityDistribution


def dynamic_k(
    entropy_val: float, vocab_size: int = 50257, k_min: int = 20, k_max: int = 500
):
    max_ent = torch.log(torch.tensor(float(vocab_size)))
    ratio = entropy_val / max_ent
    return int(k_min + ratio * (k_max - k_min))


def plot_distributions(
    d: ModalityDistribution, k_min: int, k_max: int, output_path: str
):
    distribution = d.distribution
    dc = distribution.clamp(min=1e-10)
    entropy = -(dc * dc.log()).sum(dim=-1)

    n, vocab_size = distribution.shape[0:2]
    _, axes = plt.subplots(n, 1, figsize=(16, 3 * n))
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        k = dynamic_k(
            entropy[i].item(), vocab_size=vocab_size, k_min=k_min, k_max=k_max
        )
        topk_vals, _ = torch.topk(distribution[i], k)

        label = d.index_to_modality[i]
        ax.bar(range(k), topk_vals.numpy(), width=1.0, color="steelblue", alpha=0.8)
        ax.set_title(f"{label} — entropy: {entropy[i]:.3f}, top K={k}", fontsize=11)
        ax.set_xlabel("Token rank")
        ax.set_ylabel("Probability")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")


def main(modality_file_path: str, k_min: int, k_max: int, output_path: str):
    d = ModalityDistribution.load(modality_file_path)
    plot_distributions(d, k_min=k_min, k_max=k_max, output_path=output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dist_file",
        type=str,
        help="path to the modality distribution .pt file",
    )
    parser.add_argument("--k-min", type=int, default=20, help="min topk possible")
    parser.add_argument("--k-max", type=int, default=500, help="max topk possible")
    parser.add_argument("-o", type=str, required=True, help="output image file path")

    args = parser.parse_args()
    main(args.dist_file, args.k_min, args.k_max, args.o)
