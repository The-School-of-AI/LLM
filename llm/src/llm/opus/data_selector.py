"""OpusDataSelector — composable middleware for OPUS data selection."""
from __future__ import annotations

import time
import logging
from typing import Any, Dict, Iterator, Tuple, Union

import torch
import torch.nn as nn

from .config import OpusConfig
from .ghost import GhostCollector
from .preconditioner import AdamWPreconditionerView
from .proxy import RandomInDistributionProxyProvider
from .selector import OpusSelector

logger = logging.getLogger(__name__)


class OpusDataSelector:
    """
    Composable middleware that sits between DataLoader and training loop.

    Takes a candidate batch (N samples), runs OPUS scoring via ghost factors
    and CountSketch features, then returns the best rho*N samples via
    Boltzmann selection.
    """

    def __init__(
        self,
        config: OpusConfig,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        proxy_loader: Union[Iterator, RandomInDistributionProxyProvider],
    ):
        self.config = config
        self.model = model  # unwrapped model (e.g. model_engine.module)
        self.optimizer = optimizer
        self.proxy_provider = (
            proxy_loader
            if isinstance(proxy_loader, RandomInDistributionProxyProvider)
            else None
        )
        if not isinstance(proxy_loader, RandomInDistributionProxyProvider):
            self._proxy_loader_source = proxy_loader
            self.proxy_loader = iter(proxy_loader)
        else:
            self._proxy_loader_source = None
            self.proxy_loader = None
        self.selector = OpusSelector(config)
        self.preconditioner = AdamWPreconditionerView(
            optimizer, strict_shard_only=config.strict_shard_preconditioner
        )
        self._step_count = 0

    def select_batch(
        self,
        candidate_batch: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Score candidates and return (selected_batch, metrics)."""
        cfg = self.config
        N = candidate_batch["input_ids"].shape[0]
        k_select = max(1, int(N * cfg.selection_ratio))

        # Random ablation mode — no scoring needed
        if cfg.selection_mode == "random":
            indices = torch.randperm(N)[:k_select]
            selected = {k: v[indices] for k, v in candidate_batch.items()}
            return selected, {"opus_mode": "random", "opus_selected_n": k_select}

        # OPUS scoring mode
        t0 = time.monotonic()
        input_ids = candidate_batch["input_ids"].to(device)
        score_ids = input_ids[:, : cfg.score_seq_len]

        # Get proxy batch (provider auto-resets on epoch boundary)
        if self.proxy_provider is not None:
            proxy_ids = self.proxy_provider.sample(
                device=device, k=cfg.proxy_batch_size, seq_len=cfg.score_seq_len
            )
        elif self.proxy_loader is not None:
            try:
                proxy_batch = next(self.proxy_loader)
            except StopIteration:
                logger.info("Proxy loader exhausted — resetting iterator for next epoch")
                try:
                    self.proxy_loader = iter(self._proxy_loader_source)
                    proxy_batch = next(self.proxy_loader)
                except StopIteration:
                    logger.warning("Proxy loader is empty after reset, falling back to random")
                    indices = torch.randperm(N)[:k_select]
                    return (
                        {k: v[indices] for k, v in candidate_batch.items()},
                        {"opus_mode": "fallback_random", "opus_selected_n": k_select},
                    )
            proxy_ids = proxy_batch["input_ids"][:, : cfg.score_seq_len].to(device)
        else:
            raise RuntimeError("No proxy data source configured")

        # Concatenate candidate + proxy for a single forward+backward pass
        combined_ids = torch.cat([score_ids, proxy_ids], dim=0)
        combined_x = combined_ids[:, :-1]
        combined_y = combined_ids[:, 1:]

        try:
            self.model.zero_grad(set_to_none=True)

            with GhostCollector(
                self.model,
                include_embeddings=cfg.include_embeddings,
                include_lm_head=cfg.include_lm_head,
            ) as collector:
                with torch.amp.autocast(
                    "cuda",
                    enabled=torch.cuda.is_available(),
                    dtype=torch.bfloat16,
                ):
                    h_score, _, _ = self.model(
                        combined_x,
                        next_token_ids=None,  # Explicitly disable MTP for OPUS scoring (golden + raw)
                        return_hidden=True,
                        return_memory=False,
                    )

                lm_weight = self.model.lm_head.weight
                H_dim = h_score.shape[-1]
                score_logits = torch.nn.functional.linear(
                    h_score.view(-1, H_dim), lm_weight
                )
                score_loss = torch.nn.functional.cross_entropy(
                    score_logits, combined_y.reshape(-1), ignore_index=-100
                )
                score_loss.backward()

                captures = collector.captures()
                moe_captures = collector.moe_captures()

            # Build sketch features using the real API
            n_candidates = score_ids.shape[0]
            n_proxy = proxy_ids.shape[0]

            c_feats, p_feats = self.selector.build_sketch_features(
                captures=captures,
                candidate_count=n_candidates,
                proxy_count=n_proxy,
                preconditioner=self.preconditioner,
                moe_captures=moe_captures if moe_captures else None,
            )

            lr = (
                self.optimizer.param_groups[0].get("lr", 1e-3)
                if self.optimizer.param_groups
                else 1e-3
            )

            # select() uses config.selection_ratio internally, no n_select param
            selection = self.selector.select(
                candidate_features=c_feats,
                proxy_features=p_feats,
                learning_rate=lr,
            )

            selected_idx = selection.selected_local_indices.cpu()
            self.model.zero_grad(set_to_none=True)

            elapsed_ms = (time.monotonic() - t0) * 1000
            metrics: Dict[str, Any] = {
                "opus_mode": "opus",
                "opus_alignment": selection.metrics.get("alignment", 0.0),
                "opus_redundancy": selection.metrics.get("redundancy", 0.0),
                "opus_entropy": selection.metrics.get("entropy", 0.0),
                "opus_selected_n": int(selected_idx.numel()),
                "opus_candidates_n": N,
                "opus_overhead_ms": elapsed_ms,
                "opus_used_fallback": selection.used_fallback,
            }
            if cfg.track_nonfinite_stats:
                metrics["opus_nonfinite_count"] = selection.metrics.get(
                    "nonfinite_feature_values", 0
                )

        except Exception as e:
            if cfg.fallback_random_on_error:
                logger.warning(
                    f"OPUS scoring failed ({type(e).__name__}: {e}), falling back to random",
                    exc_info=True,
                )
                selected_idx = torch.randperm(N)[:k_select]
                self.model.zero_grad(set_to_none=True)
                metrics = {
                    "opus_mode": "fallback_random",
                    "opus_error": str(e),
                    "opus_selected_n": k_select,
                }
            else:
                raise

        selected_batch = {k: v[selected_idx] for k, v in candidate_batch.items()}
        self._step_count += 1
        return selected_batch, metrics

    def refresh_preconditioner(self) -> None:
        """Refresh cached preconditioner values (call after optimizer step)."""
        self.preconditioner.refresh()
