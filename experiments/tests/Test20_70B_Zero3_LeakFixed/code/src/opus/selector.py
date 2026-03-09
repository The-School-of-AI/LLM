from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from src.opus.distributed import (
    all_gather_1d,
    all_reduce_sum_async,
    broadcast_tensor,
    get_rank,
    get_world_size,
)


@dataclass
class SelectionResult:
    # Indices into this GPU's local candidate pool.
    # Use these directly to index into candidate_ids in the training pass.
    selected_local_indices: torch.Tensor  # (k_local,)

    # Global indices encoding both owner GPU and local position:
    #   global_idx = owner_rank * n_local + local_idx
    # Useful for logging and cross-GPU analysis.
    selected_global_indices: torch.Tensor  # (k_global,)

    # True if the random fallback was triggered (zero winners or error)
    used_fallback: bool

    # Diagnostic metrics from the selection loop
    metrics: Dict[str, float]


class OpusSelector:
    """
    Iterative Boltzmann selection loop for OPUS.

    At each iteration we want to sample one candidate from the global pool
    proportional to exp(U_z / τ), where U_z is the OPUS utility score:

        U_z = η * Σ_r <φ^(r)(z), ψ^(r)_proxy>   (alignment)
            - η² * Σ_r <φ^(r)(z), Φ^(r)>          (redundancy penalty)

    Rather than all-gathering all candidate sketches (O(N * world_size * sketch_dim)
    communication per step), we use the Gumbel-max trick distributed across GPUs:

        1. Each GPU adds independent Gumbel noise to its local scores → local nominee
        2. Local nominees compete globally via a small all-gather (O(world_size))
        3. The winner's sketch is all-reduced so all GPUs update history Φ identically

    This is statistically equivalent to running Boltzmann sampling over the full
    global pool, at a fraction of the communication cost.

    Each GPU has different candidates and a different parameter shard (ZeRO-3).
    Scores are local/partial. Each GPU nominates its local Gumbel best, then
    nominations compete globally. Approximation: scores across GPUs are not on
    the same scale since each GPU only holds a shard of the optimizer state.
    """

    def __init__(
        self,
        selection_ratio: float,
        temperature: float,
        seed: int = 42,
        max_selector_time_s: float = 30.0,
        fallback_random_on_error: bool = True,
    ):
        if not (0.0 < selection_ratio <= 1.0):
            raise ValueError(
                f"selection_ratio must be in (0, 1], got {selection_ratio}"
            )
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.selection_ratio = selection_ratio
        self.temperature = max(temperature, 1e-6)
        self.max_selector_time_s = max_selector_time_s
        self.fallback_random_on_error = fallback_random_on_error

        # Each GPU uses a rank-offset seed so Gumbel noise is independent across
        # ranks — this is required for the distributed Gumbel-max trick to be
        # statistically correct (each GPU must draw its own independent noise).
        rank = get_rank()
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(seed + rank)

    # -------------------------------------------------------------------------
    # Gumbel argmax
    # -------------------------------------------------------------------------

    def _local_gumbel_argmax(
        self, logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adds Gumbel noise to logits and returns (noisy_value, argmax_index).

        Sampling from categorical(softmax(logits)) is equivalent to:
            1. Draw U_i ~ Uniform(0,1) for each logit i
            2. Compute Gumbel noise G_i = -log(-log(U_i))
            3. Return argmax(logits + G)

        Each GPU draws independent noise (different RNG seed per rank), which
        is correct — we want each GPU to independently nominate its local best
        before competing globally in _global_pick_from_rank_bests.
        """
        u = (
            torch.rand(logits.shape, generator=self._rng, dtype=torch.float32)
            .clamp_(min=1e-6, max=1.0 - 1e-6)
            .to(logits.device)
        )
        gumbel_noise = -torch.log(-torch.log(u))
        noisy = logits.float() + gumbel_noise
        best_idx = torch.argmax(noisy)
        return noisy[best_idx], best_idx

    # -------------------------------------------------------------------------
    # Global winner selection (ZeRO-3 mode)
    # -------------------------------------------------------------------------

    def _global_pick_from_rank_bests(
        self,
        local_best_value: torch.Tensor,  # scalar: Gumbel-noisy logit of local winner
        local_best_index: torch.Tensor,  # scalar: local index of local winner
        local_best_score: torch.Tensor,  # scalar: raw utility score of local winner
        n_local: int,
        device: torch.device,
    ) -> Tuple[int, float]:
        """
        Competes each GPU's local Gumbel-noisy winner to elect one global winner.

        All-gather gives every GPU the full set of (value, index, score) nominees.
        Rank 0 picks the nominee with the highest Gumbel-noisy logit, encodes the
        result as a global index, and broadcasts to all GPUs.

        Global index encoding: owner_rank * n_local + local_idx

        Returns (-1, 0.0) if no finite winner can be found.
        """
        # All-gather one scalar per GPU: cheap, O(world_size) communication
        gathered_vals = all_gather_1d(local_best_value.view(1).float())  # (world_size,)
        gathered_idx = all_gather_1d(local_best_index.view(1).long())  # (world_size,)
        gathered_score = all_gather_1d(
            local_best_score.view(1).float()
        )  # (world_size,)

        # Rank 0 picks the global winner and broadcasts
        if get_rank() == 0:
            owner = int(torch.argmax(gathered_vals).item())
            best_val = float(gathered_vals[owner].item())

            # Reject if the best noisy logit is non-finite or was a masked -inf
            if (
                not math.isfinite(best_val)
                or best_val <= torch.finfo(torch.float32).min * 0.5
            ):
                chosen = -1
                chosen_score = 0.0
            else:
                local_idx = int(gathered_idx[owner].item())
                chosen = owner * n_local + local_idx
                chosen_score = float(gathered_score[owner].item())
        else:
            chosen = -1
            chosen_score = 0.0

        # Broadcast rank 0's decision to all GPUs
        chosen_t = torch.tensor([chosen], device=device, dtype=torch.long)
        chosen_score_t = torch.tensor(
            [chosen_score], device=device, dtype=torch.float32
        )
        broadcast_tensor(chosen_t, src=0)
        broadcast_tensor(chosen_score_t, src=0)

        return int(chosen_t.item()), float(chosen_score_t.item())

    # -------------------------------------------------------------------------
    # Zero-winner fallback
    # -------------------------------------------------------------------------

    def _fallback_local_random(
        self, n_local: int, k: int, device: torch.device
    ) -> torch.Tensor:
        """
        Randomly sample k indices from [0, n_local) without replacement.

        Used when a GPU receives zero winners from the global Boltzmann loop.
        Runs independently per GPU with no cross-GPU coordination.

        NOTE: This adds k_local_fallback extra samples on top of the globally
        selected K for that step, making the total slightly larger than
        selected_batch_size * world_size. This is a known minor inconsistency
        accepted as reasonable for a rare edge case — it avoids an empty training
        step without introducing a dummy/zero-loss update.
        """
        k = min(k, n_local)
        perm = torch.randperm(n_local, generator=self._rng, device="cpu")[:k]
        return perm.to(device)

    # -------------------------------------------------------------------------
    # Main selection loop
    # -------------------------------------------------------------------------

    def select(
        self,
        alignment_scores: torch.Tensor,  # (n_local,)
        candidate_sketches: Dict[str, torch.Tensor],  # layer -> (n_local, sketch_dim)
        learning_rate: float,
    ) -> SelectionResult:
        """
        Run the iterative Boltzmann selection loop.

        Args:
            alignment_scores:
                Per-candidate alignment with proxy, already summed over layers:
                    Σ_r <φ^(r)(z), ψ^(r)_proxy>
                Pre-computed by OpusGhostCollector — no extra work needed here.

            candidate_sketches:
                Per-layer candidate sketch features φ^(r)(z), needed to update
                the redundancy history Φ each time a candidate is selected.

            learning_rate:
                The current step's learning rate. Passed explicitly each call so
                utility scores automatically track the LR schedule.
        """
        t_start = time.perf_counter()
        device = alignment_scores.device
        lr = float(learning_rate)
        n_local = alignment_scores.shape[0]
        world = get_world_size()
        rank = get_rank()

        # Stack per-layer sketches into a single tensor for einsum efficiency.
        # Sorting layer names ensures identical ordering across all GPUs.
        layer_names = sorted(candidate_sketches.keys())
        cand = torch.stack(
            [candidate_sketches[name].float() for name in layer_names], dim=1
        )  # (n_local, n_layers, sketch_dim)

        # Total number of candidates to select globally this step.
        # Each GPU has a different local pool, so the global pool is n_local * world.
        k_global = max(1, int(round(self.selection_ratio * n_local * world)))

        # Fallback k: how many random samples to draw if a GPU gets zero winners
        k_local_fallback = max(1, int(round(self.selection_ratio * n_local)))

        # History Φ^(r): running sum of selected candidates' sketches across layers.
        # Shape: (n_layers, sketch_dim). Kept identical on all GPUs via all-reduce.
        history = torch.zeros(
            cand.shape[1], cand.shape[2], device=device, dtype=torch.float32
        )

        # Track which local candidates have already been selected this step
        selected_local = torch.zeros(n_local, dtype=torch.bool, device=device)

        selected_global_indices: List[int] = []

        # Metrics
        alignment_acc = 0.0
        redundancy_acc = 0.0
        entropy_acc = 0.0
        nonfinite_scores = 0
        used_fallback = False
        fallback_no_finite = False

        for _ in range(k_global):
            # ── Timeout guard ─────────────────────────────────────────────
            if (time.perf_counter() - t_start) > self.max_selector_time_s:
                if not self.fallback_random_on_error:
                    raise TimeoutError(
                        f"OPUS selector exceeded time budget of {self.max_selector_time_s}s"
                    )
                local_indices = self._fallback_local_random(
                    n_local, k_local_fallback, device
                )
                global_indices = local_indices + (rank * n_local)
                return SelectionResult(
                    selected_local_indices=local_indices,
                    selected_global_indices=global_indices,
                    used_fallback=True,
                    metrics={
                        "alignment": 0.0,
                        "redundancy": 0.0,
                        "entropy": 0.0,
                        "nonfinite_scores": float(nonfinite_scores),
                        "fallback_no_finite": 0.0,
                        "used_fallback": 1.0,
                        "selector_time_s": float(time.perf_counter() - t_start),
                    },
                )

            # ── Compute local utility scores ───────────────────────────────
            # Alignment term: η * Σ_r <φ^(r)(z), ψ^(r)_proxy>
            # alignment_scores already holds the layer-summed dot product
            # from the ghost collector, so no further summation needed here.
            alignment_term = lr * alignment_scores.float()

            # Redundancy term: η² * Σ_r <φ^(r)(z), Φ^(r)>
            # history accumulates the sketches of already-selected candidates.
            redundancy_term = (lr**2) * torch.einsum("nlm,lm->n", cand, history)

            local_scores = alignment_term - redundancy_term

            # Clamp non-finite scores to a large negative value rather than
            # -inf so they don't propagate NaN into Gumbel noise arithmetic.
            nonfinite_mask = ~torch.isfinite(local_scores)
            nonfinite_scores += int(nonfinite_mask.sum().item())
            local_scores = torch.nan_to_num(
                local_scores,
                nan=torch.finfo(torch.float32).min,
                posinf=torch.finfo(torch.float32).min,
                neginf=torch.finfo(torch.float32).min,
            )

            # Mask already-selected candidates with -inf so they can never
            # be picked again (Gumbel noise can't rescue a -inf logit)
            local_scores[selected_local] = -torch.inf

            # ── Sample one winner ──────────────────────────────────────────

            # Each GPU picks its local Gumbel best, then they compete globally.
            logits = local_scores / self.temperature
            local_best_val, local_best_idx = self._local_gumbel_argmax(logits)
            local_best_score = local_scores[local_best_idx]

            chosen, chosen_score = self._global_pick_from_rank_bests(
                local_best_value=local_best_val,
                local_best_index=local_best_idx,
                local_best_score=local_best_score,
                n_local=n_local,
                device=device,
            )

            # ── Handle no valid winner ─────────────────────────────────────
            if chosen < 0:
                fallback_no_finite = True
                break

            selected_global_indices.append(chosen)

            # ── Update local selection mask ────────────────────────────────
            # Decode owner GPU and local index from the global index:
            #   global_idx = owner_rank * n_local + local_idx
            owner = chosen // n_local
            local_idx = chosen % n_local
            if rank == owner:
                selected_local[local_idx] = True
                selected_feat = cand[local_idx].clone()  # (n_layers, sketch_dim)
            else:
                # This GPU doesn't own the winner — contribute zeros to
                # the all-reduce so history stays correct everywhere
                selected_feat = torch.zeros_like(history)

            # ── Update history Φ on all GPUs ───────────────────────────────
            # All-reduce propagates the winner's sketch to every GPU so they
            # all update Φ identically, keeping the redundancy term in sync.
            work = all_reduce_sum_async(selected_feat)
            if work is not None:
                work.wait()
            history = history + selected_feat

            # ── Metrics ────────────────────────────────────────────────────
            alignment_acc += chosen_score
            redundancy_acc += float(torch.norm(selected_feat).item())

            # Approximate entropy contribution: -p * log(p)
            if math.isfinite(chosen_score):
                p = math.exp(min(chosen_score / self.temperature, 80.0))
                if p > 0.0:
                    entropy_acc += -p * math.log(p + 1e-12)

        # ── Post-loop: resolve final local indices ─────────────────────────────
        local_indices = torch.nonzero(selected_local, as_tuple=False).flatten()

        if local_indices.numel() == 0:
            # This GPU received zero winners from the global competition.
            # Fall back to a random selection from the local pool to avoid an
            # empty training step.
            #
            # NOTE: This is a known minor inconsistency — see _fallback_local_random
            # docstring and train.py for full discussion.
            local_indices = self._fallback_local_random(
                n_local, k_local_fallback, device
            )
            used_fallback = True

        # Build global index tensor from whatever local indices we ended up with
        if selected_global_indices:
            global_idx_tensor = torch.tensor(
                selected_global_indices, device=device, dtype=torch.long
            )
        else:
            # Only happens if fallback triggered before any winner was recorded
            global_idx_tensor = local_indices + (rank * n_local)

        return SelectionResult(
            selected_local_indices=local_indices,
            selected_global_indices=global_idx_tensor,
            used_fallback=used_fallback,
            metrics={
                "alignment": alignment_acc,
                "redundancy": redundancy_acc,
                "entropy": entropy_acc,
                "nonfinite_scores": float(nonfinite_scores),
                "fallback_no_finite": 1.0 if fallback_no_finite else 0.0,
                "used_fallback": 1.0 if used_fallback else 0.0,
                "selector_time_s": float(time.perf_counter() - t_start),
            },
        )
