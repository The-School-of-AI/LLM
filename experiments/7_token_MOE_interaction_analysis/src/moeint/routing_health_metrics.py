from enum import IntEnum
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor


class TokenGroups(IntEnum):
    whitespace = 0  # for now, considered as junk
    content = 1


class NullExpertStats(NamedTuple):
    junk_to_null_rate: float
    """
    Out of the junk tokens present in the batch, how many of them went to null expert?

    Tells us whether we are successfully routing junk tokens present in batch to null experts.
    Value will be NaN if there are no junk tokens present in the batch.
    """

    null_junk_rate: float
    """
    Out of the tokens sent to null expert, how many of them were junk?

    Tells us whether the null experts are staying pure, or if they are eating content related tokens too.
    Value will be NaN if no tokens were sent to null expert
    """

    tokens_to_null_rate: float
    """
    How many tokens in the batch were sent to null experts?
    """


class NullRoutingAnalyzer:
    def __init__(
        self,
        vocab_size: int,
        token_id_group_mapping: dict[int, int],
        device: str = "cpu",
    ):
        # pre-allocate a mapping for the entire vocab.
        # default token group for all tokens is "content"
        self.group_map = torch.full(
            (vocab_size,), TokenGroups.content, dtype=torch.long, device=device
        )

        # update the group map
        for tid, gid in token_id_group_mapping.items():
            self.group_map[tid] = gid

    def analyze(
        self,
        input_ids: Tensor,  # (batch_size, seq_len)
        routing_logits: list[
            Tensor
        ],  # list of tensor of shape (batch_size * seq_len, num_real_experts + num_null_experts), one for each moe block
        num_real_experts: int,
        topk: int,
    ) -> list[NullExpertStats]:
        input_group_ids = self._token_ids_to_token_groups(input_ids).view(
            -1
        )  # (batch_size * seq_len)
        null_expert_stats: list[NullExpertStats] = []

        for rl in routing_logits:
            routing_probs = torch.sigmoid(rl)
            _, indices = torch.topk(
                routing_probs, topk, dim=-1
            )  # indices is of shape (batch_size * seq_len, num_real_experts + num_null_experts)

            tokens_routed_to_null = torch.any(
                indices
                >= num_real_experts,  # assuming last num_null_experts out of num_real_experts + num_null_experts are null experts
                dim=1,
            )  # bool tensor of (batch_size * seq_len,). if value at index i is True, that means ith token was routed to null expert.
            tokens_to_null_rate = (
                tokens_routed_to_null.sum().item() / tokens_routed_to_null.numel()
            )

            groups_routed_to_null = input_group_ids[tokens_routed_to_null]
            if groups_routed_to_null.numel() == 0:
                null_junk_rate = float("nan")
            else:
                null_junk_rate = (
                    torch.sum(groups_routed_to_null == TokenGroups.whitespace)
                    / groups_routed_to_null.numel()
                ).item()

            is_junk_token = input_group_ids == TokenGroups.whitespace
            if is_junk_token.sum() == 0:
                junk_to_null_rate = float("nan")
            else:
                junk_sent_to_null = torch.logical_and(
                    is_junk_token, tokens_routed_to_null
                )
                junk_to_null_rate = (
                    torch.sum(junk_sent_to_null) / torch.sum(is_junk_token).item()
                ).item()

            null_expert_stats.append(
                NullExpertStats(
                    null_junk_rate=null_junk_rate,
                    junk_to_null_rate=junk_to_null_rate,
                    tokens_to_null_rate=tokens_to_null_rate,
                )
            )

        return null_expert_stats

    def _token_ids_to_token_groups(self, input_ids: Tensor) -> Tensor:
        """
        Assigns group ids to each token id present in the input.

        Args:
            input_ids (Tensor): Tensor of shape (batch_size, seq_len) containing token ids

        Returns:
            Tensor: Tensor of shape (batch_size, seq_len) containing token group ids
        """

        return self.group_map[input_ids]


class RouterLoadDistributionStats(NamedTuple):
    tokens_per_expert: Tensor

    @property
    def maxmin(self) -> float:
        """
        Ratio between the busiest expert and the least busy expert.

        Lower value -> more balanced.
        """
        return (
            self.tokens_per_expert.max() / max(self.tokens_per_expert.min(), 1)
        ).item()

    @property
    def gini(self) -> float:
        """
        The Gini Coefficient.

        Measure of inequality in workload distribution between experts.
        """
        arr, _ = torch.sort(self.tokens_per_expert)
        n = arr.numel()
        index = torch.range(1, n + 1)

        return (
            (2 * torch.sum(index * arr)) / (n * torch.sum(arr)) - (n + 1) / n
        ).item()

    @property
    def cv(self):
        """
        Coefficient of variation.

        Measure of how spread out the workload among experts is, as a percentage.
        """

        mean = self.tokens_per_expert.float().mean()
        if mean == 0:
            return float("nan")

        std = self.tokens_per_expert.float().std()
        return (std / mean).item()


class RouterStats(NamedTuple):
    load_stats: list[RouterLoadDistributionStats]
    """
    Load distribution statistics for each moe block.
    """

    entropy_per_token: list[Tensor]
    """
    Per token entropy of the router in each moe block.
    """


class RouterAnalyzer:
    def analyze(
        self,
        routing_logits: list[
            Tensor
        ],  # list of tensor of shape (batch_size * seq_len, num_real_experts + num_null_experts), one for each moe block
        num_real_experts: int,
        num_null_experts: int,
        topk: int,
    ) -> RouterStats:
        load_stats: list[RouterLoadDistributionStats] = []
        entropy_per_token: list[Tensor] = []

        for rl in routing_logits:
            routing_probs = torch.sigmoid(rl)
            _, indices = torch.topk(
                routing_probs, topk, dim=-1
            )  # indices is of shape (batch_size * seq_len, num_real_experts + num_null_experts)

            load_stats.append(
                self._calculate_load_distribution(
                    indices, num_real_experts, num_null_experts
                )
            )

            entropy_per_token.append(self._calculate_entropy_per_token(rl))

        return RouterStats(load_stats, entropy_per_token)

    def _calculate_load_distribution(
        self, topk_indices: Tensor, num_real_experts: int, num_null_experts: int
    ) -> RouterLoadDistributionStats:
        num_total_experts = num_real_experts + num_null_experts

        tokens_per_expert = torch.bincount(
            topk_indices.flatten(), minlength=num_total_experts
        )
        tokens_per_null_expert = tokens_per_expert[num_real_experts:]
        tokens_per_expert = torch.cat(
            [
                tokens_per_expert[:num_real_experts],
                tokens_per_null_expert.sum().unsqueeze(0),
            ]
        )  # treating all null experts as one expert

        return RouterLoadDistributionStats(tokens_per_expert)

    def _calculate_entropy_per_token(self, routing_logits: Tensor) -> Tensor:
        log_probs = F.log_softmax(routing_logits)
        probs = F.softmax(routing_logits)
        return -torch.sum(probs * log_probs)


class RouterHealthMetrics(NamedTuple):
    null_experts_stats: list[NullExpertStats]
    router_stats: RouterStats


class RouterHealthAnalyzer:
    def __init__(
        self,
        vocab_size: int,
        token_id_group_mapping: dict[int, int],
        num_real_experts: int,
        num_null_experts: int,
        topk: int,
        device: str = "cpu",
    ):
        self.num_real_experts = num_real_experts
        self.num_null_experts = num_null_experts
        self.topk = topk

        self.null_routing_analyzer = NullRoutingAnalyzer(
            vocab_size, token_id_group_mapping, device
        )
        self.router_analyzer = RouterAnalyzer()

    def analyze(
        self, input_ids: Tensor, routing_logits: list[Tensor]
    ) -> RouterHealthMetrics:
        null_experts_stats = self.null_routing_analyzer.analyze(
            input_ids, routing_logits, self.num_real_experts, self.topk
        )
        router_stats = self.router_analyzer.analyze(
            routing_logits, self.num_real_experts, self.num_null_experts, self.topk
        )

        return RouterHealthMetrics(null_experts_stats, router_stats)
