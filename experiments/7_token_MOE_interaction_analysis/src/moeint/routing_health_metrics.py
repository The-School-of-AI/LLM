from enum import IntEnum
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor


class TokenGroups(IntEnum):
    junk = 0
    content = 1


class RouterStats(NamedTuple):
    entropy_mean: float
    """
    The mean entropy for this router.

    Measure of how confident the router was when routing a token.
    Low entropy -> more confident, high entropy -> less confident.
    """

    entropy_std: float
    """
    Standard deviation of entropy, i.e., its spread.
    """

    entropy_per_group: list[float]
    """
    Average entropy for each token group for this router.

    Value at index i is the average entropy of the ith group.
    """

    n_eff: float
    """
    Effective number of experts being used (out of num_real_experts + 1).

    Calculated as exp(entropy_mean). Includes null expert collapsed as single option.
    Values close to (num_real_experts + 1) indicate uniform routing, close to 1 indicate concentration.
    """

    polarization: float
    """
    Measure of bimodality: are tokens getting routed to only real or mostly null experts?

    Value near 0 = gradual distribution, near 1 = polarized.
    """

    tokens_to_real_rate: float
    """
    What fraction of tokens in the batch were sent to real experts?
    Value between 0.0 and 1.0.
    """

    tokens_to_null_rate: float
    """
    What fraction of tokens in the batch were sent to null experts?
    Value between 0.0 and 1.0.
    """

    null_junk_rate: float
    """
    Out of the tokens sent to null expert, what fraction of them were junk?

    Tells us whether the null experts are staying pure, or if they are also receiving content related tokens too.
    Value will be NaN if no tokens were sent to null expert
    """

    junk_to_null_rate: float
    """
    Out of the junk tokens present in the batch, what fraction went to null experts?

    Tells us whether we are successfully routing junk tokens to null experts.
    Value will be NaN if there are no junk tokens present in the batch.
    """

    imbalance_ratio: float
    """
    Ratio between the busiest real expert and the least busy real expert.

    Lower value -> more balanced.
    Null experts are ignored.
    """

    gini: float
    """
    The Gini Coefficient.

    Measure of inequality in workload distribution between real experts.
    """

    cv: float
    """
    Coefficient of variation.

    Measure of how spread out the workload is among real experts.
    """

    starvation_count: int
    """
    Number of real experts receiving less than ε traffic.

    Expert is starving if it receives tokens < ε * average tokens per expert.
    """

    router_logit_scale: float
    """
    Standard deviation of router logits.
    """


class RouterHealthAnalyzer:
    def __init__(
        self,
        vocab_size: int,
        token_id_group_mapping: dict[int, int],
        num_experts: int,
        data_sparsity: float,
        topk: int,
        starvation_threshold: float = 0.01,
        polarization_threshold: tuple[float, float] = (0.1, 0.9),
        device: str = "cpu",
    ):
        if polarization_threshold[0] > polarization_threshold[1]:
            raise ValueError("polarization threshold min is greater than max")

        self.num_real_experts = num_experts
        self.num_null_experts = int(
            self.num_real_experts * (1 - data_sparsity) / data_sparsity
        )
        self.topk = topk
        self.starvation_threshold = starvation_threshold
        self.polarization_threshold = polarization_threshold
        self.group_map = torch.full(
            (vocab_size,), TokenGroups.content, dtype=torch.long, device=device
        )
        for tid, gid in token_id_group_mapping.items():
            self.group_map[tid] = gid

    def analyze_logits(
        self, input_ids: Tensor, routing_logits: list[Tensor]
    ) -> list[RouterStats]:
        """
        Collects statistics from router logits.

        Args:
            input_ids (Tensor): A tensor containing input ids of each token in the batch. Shape: (batch_size, seq_len)
            routing_logits (list[Tensor]): A list containing each router's logits. Shape: (batch_size, seq_len, num_real_experts + num_null_experts)

        Returns:
            list[RouterStats]: stats for each router
        """

        num_routers = len(routing_logits)
        num_tokens = input_ids.numel()
        routing_logits_tensor = torch.stack(routing_logits).flatten(
            start_dim=1, end_dim=2
        )  # shape: (num_routers, batch_size * seq_len, num_real_experts + num_null_experts)
        input_group_ids = (
            self.group_map[input_ids]
            .flatten()  # shape: (batch_size * seq_len,)
            .unsqueeze(0)
            .expand(
                num_routers, num_tokens
            )  # shape: (num_routers, batch_size * seq_len)
        )  # group ids of each token present in input_ids. the group ids are same across dim=0

        router_logits_scale = routing_logits_tensor.std(dim=(-2, -1))
        entropy_per_token = self._calculate_entropy_per_token(routing_logits_tensor)
        entropy_per_group = self._calculate_entropy_per_group(
            entropy_per_token, input_group_ids
        )
        entropy_mean = entropy_per_token.mean(dim=1)  # shape: (num_routers,)
        entropy_std = entropy_per_token.std(dim=1)  # shape: (num_routers,)
        n_eff = torch.exp(entropy_mean)  # shape: (num_routers,)

        routing_probs = F.softmax(
            routing_logits_tensor, dim=-1
        )  # shape: (num_routers, batch_size * seq_len, num_real_experts + num_null_experts)
        polarization = self._calculate_polarization(routing_probs)

        _, topk_indices = torch.topk(
            routing_probs, self.topk, dim=-1
        )  # topk_indices shape: (num_routers, batch_size * seq_len, topk)
        tokens_to_real_rate = self._calculate_tokens_to_real_rate(topk_indices)
        tokens_to_null_rate = self._calculate_tokens_to_null_rate(topk_indices)
        null_junk_rate = self._calculate_null_junk_rate(topk_indices, input_group_ids)
        junk_to_null_rate = self._calculate_junk_to_null_rate(
            topk_indices, input_group_ids
        )

        tokens_per_expert = self._calculate_tokens_per_expert(topk_indices)
        imbalance_ratio = self._calculate_imbalance_ratio(tokens_per_expert)
        gini = self._calculate_gini(tokens_per_expert)
        cv = self._calculate_cv(tokens_per_expert)
        starvation_count = self._calculate_starvation_count(tokens_per_expert)

        router_stats: list[RouterStats] = []
        for (
            entropy_mean,
            entropy_std,
            entropy_per_group,
            n_eff,
            polarization,
            tokens_to_real_rate,
            tokens_to_null_rate,
            null_junk_rate,
            junk_to_null_rate,
            imbalance_ratio,
            gini,
            cv,
            starvation_count,
            router_logits_scale,
        ) in zip(
            entropy_mean,
            entropy_std,
            entropy_per_group,
            n_eff,
            polarization,
            tokens_to_real_rate,
            tokens_to_null_rate,
            null_junk_rate,
            junk_to_null_rate,
            imbalance_ratio,
            gini,
            cv,
            starvation_count,
            router_logits_scale,
        ):
            router_stats.append(
                RouterStats(
                    entropy_mean=entropy_mean.item(),
                    entropy_std=entropy_std.item(),
                    entropy_per_group=entropy_per_group.cpu().tolist(),
                    n_eff=n_eff.item(),
                    polarization=polarization.item(),
                    tokens_to_real_rate=tokens_to_real_rate.item(),
                    tokens_to_null_rate=tokens_to_null_rate.item(),
                    null_junk_rate=null_junk_rate.item(),
                    junk_to_null_rate=junk_to_null_rate.item(),
                    imbalance_ratio=imbalance_ratio.item(),
                    gini=gini.item(),
                    cv=cv.item(),
                    starvation_count=starvation_count.item(),
                    router_logit_scale=router_logits_scale.item(),
                )
            )

        return router_stats

    def _calculate_entropy_per_token(self, routing_logits: Tensor) -> Tensor:
        """
        Calculates entropy per token.

        Args:
            routing_logits (Tensor): A tensor of shape (num_routers, batch_size * seq_len, num_real_experts + num_null_experts)

        Returns:
            Tensor: A tensor of shape (num_routers, batch_size * seq_len), the entropy per token for each router, where all null
            experts are considered as a single expert
        """

        probs = F.softmax(routing_logits, dim=-1)
        real_probs = probs[..., : self.num_real_experts]
        null_prob = probs[..., self.num_real_experts :].sum(dim=-1, keepdim=True)
        collapsed_probs = torch.cat([real_probs, null_prob], dim=-1)

        log_probs = torch.log(collapsed_probs + 1e-10)
        return -torch.sum(collapsed_probs * log_probs, dim=-1)

    def _calculate_tokens_per_expert(self, topk_indices: Tensor) -> Tensor:
        """
        Calculates tokens per expert. All null experts are treated as a single expert.

        Args:
            topk_indices (Tensor): A tensor of shape (num_routers, batch_size * seq_len, topk), in which the value at [i, j] are the experts
            that token j of router i was sent to.

         Returns:
             Tensor: The tokens per expert. Tensor shape: (num_routers, num_real_experts + 1)
        """

        num_total_experts = self.num_real_experts + self.num_null_experts
        num_routers, num_tokens, topk = topk_indices.shape

        indices_flat = topk_indices.flatten(start_dim=1)
        counts = torch.zeros(
            num_routers, num_total_experts, dtype=torch.long, device=topk_indices.device
        )
        ones = torch.ones_like(indices_flat, dtype=torch.long)
        counts.scatter_add_(1, indices_flat, ones)
        real_counts = counts[:, : self.num_real_experts]
        null_counts = counts[:, self.num_real_experts :].sum(dim=1, keepdim=True)

        return torch.cat([real_counts, null_counts], dim=1)

    def _calculate_entropy_per_group(
        self, entropy_per_token: Tensor, input_group_ids: Tensor
    ) -> Tensor:
        """
        Calculates entropy per token group, give the entropies for each token.

        Args:
            entropy_per_token (Tensor): A tensor of shape (num_routers, batch_size * seq_len), the entropy per token for each router, where all null experts are considered as a single expert.
            input_group_ids (Tensor): Group ids of each token present in input batch. The value at (i, j) is the group number of router i and token j. Tensor shape: (num_routers, batch_size * seq_len)
        Returns:
            Tensor: A tensor of shape (num_routers, num_groups), representing the entropy per group per router. If a group was not present in the input batch, the entropy is NaN.
        """

        num_groups = len(TokenGroups)  # Total possible groups, not just in batch

        # Convert group IDs to one-hot vectors
        # Each token gets a vector like [0, 1, 0] indicating which group it belongs to
        # Shape: (num_routers, num_tokens, num_groups)
        # Example: token with group_id=1 becomes [0, 1, 0] for 3 groups
        group_mask = F.one_hot(input_group_ids, num_classes=num_groups).float()

        # Prepare entropy for broadcasting
        # Add a dimension so we can multiply with group_mask
        # Shape: (num_routers, num_tokens, 1) -> broadcasts to (num_routers, num_tokens, num_groups)
        entropy_expanded = entropy_per_token.unsqueeze(-1)

        # Route each token's entropy to its group position
        # Multiply: entropy value gets placed in the group's position, zeros elsewhere
        # Example: token with entropy=2.5 and group_id=1 produces [0, 2.5, 0]
        # Then sum across all tokens to aggregate entropy per group
        # Result shape: (num_routers, num_groups)
        group_sum = (entropy_expanded * group_mask).sum(dim=1)

        # Count how many tokens belong to each group
        # Sum the one-hot vectors across tokens dimension
        # Result shape: (num_routers, num_groups)
        # Example: if 5 tokens have group_id=0, group_count[..., 0] = 5
        group_count = group_mask.sum(dim=1)

        # Calculate mean entropy per group, replace inf with NaN as per contract.
        # Result shape: (num_routers, num_groups)
        group_entropy = group_sum / group_count
        group_entropy.masked_fill_(torch.isinf(group_entropy), float("nan"))

        return group_entropy

    def _calculate_imbalance_ratio(self, tokens_per_expert: Tensor) -> Tensor:
        """
        Calculates the imbalance ratio of real experts.

        Args:
            tokens_per_expert (Tensor): The tokens per expert. Tensor shape: (num_routers, num_real_experts + 1)

        Returns:
            Tensor: The imbalance ratio for each router. Tensor shape: (num_routers,)
        """

        tokens_per_real_expert = tokens_per_expert[:, :-1]
        return torch.max(tokens_per_real_expert, dim=-1).values / torch.clamp(
            torch.min(tokens_per_real_expert, dim=-1).values, min=1
        )

    def _calculate_gini(self, tokens_per_expert: Tensor) -> Tensor:
        """
        Calculates the Gini coefficient of real experts.

        Args:
            tokens_per_expert (Tensor): The tokens per expert. Tensor shape: (num_routers, num_real_experts + 1)

        Returns:
            Tensor: The Gini coefficients for each router. Tensor shape: (num_routers,)
        """

        tokens_per_real_expert = tokens_per_expert[:, :-1]
        arr, _ = torch.sort(tokens_per_real_expert, dim=-1)
        n = tokens_per_real_expert.shape[-1]
        index = torch.arange(
            1, n + 1, device=tokens_per_real_expert.device, dtype=torch.float
        )
        return (2 * torch.sum(index * arr, dim=-1)) / (n * torch.sum(arr, dim=-1)) - (
            n + 1
        ) / n

    def _calculate_cv(self, tokens_per_expert: Tensor) -> Tensor:
        """
        Calculates the coefficient of variation (CV) of real experts.

        Args:
            tokens_per_expert (Tensor): The tokens per expert. Tensor shape: (num_routers, num_real_experts + 1)

        Returns:
            Tensor: The CVs for each router. Tensor shape: (num_routers,)
        """

        tokens_per_real_expert = tokens_per_expert[:, :-1].float()
        return tokens_per_real_expert.std(dim=-1) / tokens_per_real_expert.mean(dim=-1)

    def _calculate_starvation_count(self, tokens_per_expert: Tensor) -> Tensor:
        """
        Calculates the number of starved real experts per router.

        Args:
            tokens_per_expert (Tensor): The tokens per expert. Tensor shape: (num_routers, num_real_experts + 1)

        Returns:
            Tensor: The number of starved for each router. Tensor shape: (num_routers,)
        """

        tokens_per_real_expert = tokens_per_expert[:, :-1].float()
        is_starving = tokens_per_real_expert < (
            self.starvation_threshold
            * tokens_per_real_expert.mean(dim=-1, keepdim=True)
        )
        return is_starving.sum(dim=-1)

    def _calculate_polarization(self, routing_probs: Tensor) -> Tensor:
        """
        Measure bimodality: tokens getting all-real vs mostly-null compute.

        Args:
            routing_probs: The routing probablities each token. Tensor shape: (num_routers, batch_size * seq_len, num_real_experts + num_null_experts)

        Returns:
            Tensor: Polarization score per router. 0 = gradual distribution, 1 = highly polarized. A Tensor of shape (num_routers,)
        """
        real_probs = routing_probs[..., : self.num_real_experts]
        compute_intensity = real_probs.sum(
            dim=-1
        )  # shape: (num_routers, batch_size * seq_len)

        near_zero = (compute_intensity < self.polarization_threshold[0]).float()
        near_one = (compute_intensity > self.polarization_threshold[1]).float()

        polarization = (near_zero.mean(dim=-1) + near_one.mean(dim=-1)) / 2

        return polarization

    def _calculate_tokens_to_real_rate(self, topk_indices: Tensor) -> Tensor:
        """
        Calculates the fraction of tokens sent to real experts.

        Args:
            topk_indices (Tensor): A tensor of shape (num_routers, batch_size * seq_len, topk), in which the value at [i, j] are the experts that token j of router i was sent to.

        Returns:
            Tensor: Fraction of tokens sent to real experts. Tensor shape: (num_routers,)
        """

        is_real = topk_indices < self.num_real_experts
        num_tokens = topk_indices.shape[1]
        tokens_routed_to_real = torch.any(
            is_real, dim=-1
        )  # bool mask, shape: (num_routers, batch_size * seq_len)
        return tokens_routed_to_real.sum(dim=1) / num_tokens

    def _calculate_tokens_to_null_rate(self, topk_indices: Tensor) -> Tensor:
        """
        Calculates the fraction of tokens sent to null experts.

        Args:
            topk_indices (Tensor): A tensor of shape (num_routers, batch_size * seq_len, topk), in which the value at [i, j] are the experts that token j of router i was sent to.

        Returns:
            Tensor: Fraction of tokens sent to null experts. Tensor shape: (num_routers,)
        """

        is_null = topk_indices >= self.num_real_experts
        num_tokens = topk_indices.shape[1]
        tokens_routed_to_null = torch.any(
            is_null, dim=-1
        )  # bool mask, shape: (num_routers, batch_size * seq_len)
        return tokens_routed_to_null.sum(dim=1) / num_tokens

    def _calculate_null_junk_rate(
        self, topk_indices: Tensor, input_group_ids: Tensor
    ) -> Tensor:
        """
        Calculate what fraction of tokens sent to null experts were junk.

        Args:
            topk_indices (Tensor): A tensor of shape (num_routers, batch_size * seq_len, topk), in which the value at [i, j] are the experts that token j of router i was sent to.
            input_group_ids (Tensor): Group ids of each token present in input batch. Tensor shape: (num_routers, batch_size * seq_len)

        Returns:
            Tensor: The null junk rate. Tensor shape: (num_routers,)
        """

        is_null = topk_indices >= self.num_real_experts
        tokens_routed_to_null = torch.any(
            is_null, dim=-1
        )  # bool mask, shape: (num_routers, batch_size * seq_len)

        is_junk = (
            input_group_ids == TokenGroups.junk
        )  # bool mask, shape: (num_routers, batch_size * seq_len)
        is_junk_and_sent_to_null = (
            tokens_routed_to_null & is_junk
        )  # shape: (num_routers, batch_size * seq_len)
        null_junk_rate = (
            is_junk_and_sent_to_null.sum(dim=1).float()
            / tokens_routed_to_null.sum(dim=1).float()
        )
        null_junk_rate.masked_fill_(torch.isinf(null_junk_rate), float("nan"))

        return null_junk_rate

    def _calculate_junk_to_null_rate(
        self, topk_indices: Tensor, input_group_ids: Tensor
    ) -> Tensor:
        """
        Calculate what fraction of junk tokens were routed to null experts.

        Args:
            topk_indices (Tensor): A tensor of shape (num_routers, batch_size * seq_len, topk), in which the value at [i, j] are the experts that token j of router i was sent to.
            input_group_ids (Tensor): Group ids of each token present in input batch. Tensor shape: (num_routers, batch_size * seq_len)

        Returns:
            Tensor: The null junk rate. Tensor shape: (num_routers,)
        """
        is_null = topk_indices >= self.num_real_experts
        tokens_routed_to_null = torch.any(
            is_null, dim=-1
        )  # bool mask, shape: (num_routers, batch_size * seq_len)
        is_junk = (
            input_group_ids == TokenGroups.junk
        )  # bool mask, shape: (num_routers, batch_size * seq_len)
        is_junk_and_sent_to_null = (
            tokens_routed_to_null & is_junk
        )  # shape: (num_routers, batch_size * seq_len)
        junk_to_null_rate = (
            is_junk_and_sent_to_null.sum(dim=1).float() / is_junk.sum(dim=1).float()
        )
        junk_to_null_rate.masked_fill_(torch.isinf(junk_to_null_rate), float("nan"))

        return junk_to_null_rate
