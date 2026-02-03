from enum import IntEnum
from typing import NamedTuple

import torch
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


class BatchStats(NamedTuple):
    null_expert_stats: list[NullExpertStats]


class RoutingAnalyzer:
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
        top_k: int,
    ) -> BatchStats:
        input_group_ids = self._token_ids_to_token_groups(input_ids).view(
            -1
        )  # (batch_size * seq_len)
        null_expert_stats: list[NullExpertStats] = []

        for rl in routing_logits:
            routing_probs = torch.sigmoid(rl)
            _, indices = torch.topk(
                routing_probs, top_k, dim=-1
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
            if is_junk_token.numel() == 0:
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

        return BatchStats(null_expert_stats=null_expert_stats)

    def _token_ids_to_token_groups(self, input_ids: Tensor) -> Tensor:
        """
        Assigns group ids to each token id present in the input.

        Args:
            input_ids (Tensor): Tensor of shape (batch_size, seq_len) containing token ids

        Returns:
            Tensor: Tensor of shape (batch_size, seq_len) containing token group ids
        """

        return self.group_map[input_ids]
