# Router Design and Formula

In a Mixture of Experts (MoE) architecture, the router (or gating network) is responsible for determining which experts should process a given input token. The router's performance is critical to the model's overall efficiency and effectiveness.

## 1. Gating Mechanism

The core of the router is the gating function $G(x)$, which produces a sparse vector of weights for the experts. For an input token $x$, the gating function is defined as:

$$H(x) = W_r \cdot x$$
$$G(x) = \text{Softmax}(\text{TopK}(H(x) + \text{Noise}, K))$$

Where:
- $W_r$: The router's weight matrix.
- $\text{Noise}$: A tunable noise term (e.g., Gaussian) to encourage exploration during training.
- $K$: The number of experts to activate for each token (typically 1 or 2).

## 2. Expert Selection

The `TopK` function selects the indices of the $K$ experts with the highest scores from $H(x)$. Let $I$ be the set of selected indices:

$$I = \text{arg-top-k}(H(x) + \text{Noise}, K)$$

The output of the MoE layer is then the weighted sum of the outputs from the selected experts:

$$y = \sum_{i \in I} G(x)_i \cdot E_i(x)$$

Where $E_i(x)$ is the output of the $i$-th expert.

---

## 3. Evolutionary Roadmap: 3/8/70 Scaling

Our roadmap scales the model's intelligence by increasing both total parameters and expert specialization:

* **Phase 1: 1B Dense Seed** - A unified network to build foundational semantic knowledge.
* **Phase 2: 3B MoE Upcycling** - We expand the 1B seed into 8 experts ($K=2$).
* **Phase 3: 8B Dense Consolidation** - Distilling MoE knowledge into a larger dense backbone for better reasoning.
* **Phase 4: 70B MoE Flagship** - Final expansion to a massive sparse model. We use **Router Upcycling** by tiling weights from Phase 2 to warm-start the 70B gating logic.

## 4. Context-Aware Routing (Bucketing Strategy)

To handle variable context lengths, the router adjusts its "confidence" via **Length-Dependent Temperature Scaling**. This encourages exploration on short sequences and specialization on long ones.

$$G(x, L) = \text{Softmax}\left(\frac{H(x) + \text{Noise}}{T(L)}\right)$$

| Bucket Length | Strategy | Target Temperature ($T$) |
| :--- | :--- | :--- |
| **1024** | **Exploration** | High ($T \approx 1.5$) |
| **2048** | **Standard** | Balanced ($T = 1.0$) |
| **4096+** | **Specialization** | Low ($T \approx 0.7$) |

## 5. Load Balancing in Packed Sequences

When using **Sequence Packing**, we implement a **Bucket-Level Auxiliary Loss** to prevent expert starvation within mixed batches.

$$\mathcal{L}_{aux} = \alpha \cdot N \sum_{b \in \text{Buckets}} \sum_{i=1}^{N} f_{i,b} \cdot P_{i,b}$$

- **$f_{i,b}$**: The fraction of tokens in bucket $b$ sent to expert $i$.
- **$P_{i,b}$**: The mean probability assigned to expert $i$ for tokens within bucket $b$.

---

## 6. Sample Implementation (PyTorch)

Below is a simplified implementation of a context-aware router with temperature scaling and load balancing support.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextAwareRouter(nn.Module):
    def __init__(self, d_model, n_experts, k=2):
        super().__init__()
        self.k = k
        self.n_experts = n_experts
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        
    def forward(self, x, bucket_len):
        # 1. Temperature selection based on bucket
        if bucket_len <= 1024:
            temp = 1.5
        elif bucket_len <= 2048:
            temp = 1.0
        else:
            temp = 0.7
            
        # 2. Compute logits with noise for exploration
        logits = self.gate(x)
        if self.training:
            noise = torch.randn_like(logits) * (1.0 / self.n_experts)
            logits = logits + noise
            
        # 3. Apply Temperature and Top-K
        logits = logits / temp
        scores, indices = torch.topk(logits, self.k, dim=-1)
        weights = F.softmax(scores, dim=-1)
        
        return weights, indices, F.softmax(logits, dim=-1)

def compute_aux_loss(probs, indices, n_experts):
    """
    Standard load balancing loss (Switch Transformer style)
    probs: raw softmax probabilities from the router
    indices: top-k indices selected
    """
    # f_i: fraction of tokens dispatched to expert i
    # (Simplified for single-bucket demonstration)
    tokens_per_expert = torch.histc(indices.float(), bins=n_experts, min=0, max=n_experts-1)
    f_i = tokens_per_expert / indices.numel()
    
    # P_i: average probability assigned to expert i
    P_i = probs.mean(dim=0)
    
    return n_experts * torch.sum(f_i * P_i)