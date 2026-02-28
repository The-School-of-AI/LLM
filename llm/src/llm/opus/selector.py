from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import torch

from .config import OpusConfig
from .distributed import (
    get_rank,
    get_world_size,
    all_gather_1d,
    broadcast_tensor,
    all_reduce_sum,
    all_reduce_sum_async,
    all_reduce_max,
)
from .countsketch import CountSketchProjector
from .ghost import LayerCapture, MoERoutedCapture
from .preconditioner import AdamWPreconditionerView

logger = logging.getLogger(__name__)


@dataclass
class SelectionResult:
    selected_local_indices: torch.Tensor
    selected_global_indices: torch.Tensor
    used_fallback: bool
    metrics: Dict[str, float]


class OpusSelector:
    """
    AdamW-only OPUS selector using ghost factors + CountSketch features.
    """

    def __init__(self, config: OpusConfig):
        self.config = config
        self.projector = CountSketchProjector(
            sketch_dim=config.sketch_dim,
            seed=config.sketch_seed,
        )
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(int(config.sketch_seed))
        self._last_feature_stats: Dict[str, float] = {}
        self._track_nonfinite = bool(getattr(config, "track_nonfinite_stats", True))

    def state_dict(self) -> Dict[str, Any]:
        return {
            "rng_state": self._rng.get_state(),
            "config": {
                "candidate_multiplier": self.config.candidate_multiplier,
                "selection_ratio": self.config.selection_ratio,
                "score_seq_len": self.config.score_seq_len,
                "proxy_batch_size": self.config.proxy_batch_size,
                "sketch_dim": self.config.sketch_dim,
                "temperature": self.config.temperature,
                "sketch_seed": self.config.sketch_seed,
            },
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if "rng_state" in state:
            self._rng.set_state(state["rng_state"])

    @staticmethod
    def _ensure_btd(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return x
        if x.dim() == 2:
            return x.unsqueeze(1)
        raise ValueError(f"Expected dim 2/3 tensor, got shape {tuple(x.shape)}")

    def _sanitize_for_scoring(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        bad = int((~torch.isfinite(x)).sum().item()) if self._track_nonfinite else 0
        x = torch.nan_to_num(x.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return x, bad

    @staticmethod
    def _silu_prime(x: torch.Tensor) -> torch.Tensor:
        sig = torch.sigmoid(x)
        return sig * (1.0 + x * (1.0 - sig))

    def _build_moe_features(
        self,
        moe_captures: Dict[str, MoERoutedCapture],
        candidate_count: int,
        proxy_count: int,
        preconditioner: AdamWPreconditionerView,
        out_dtype: torch.dtype,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, float]]:
        candidate_features: Dict[str, torch.Tensor] = {}
        proxy_features: Dict[str, torch.Tensor] = {}

        total_bad_act = 0.0
        total_bad_grad = 0.0
        total_bad_precond = 0.0

        for layer_name, cap in list(moe_captures.items()):
            x_all, bad_x = self._sanitize_for_scoring(self._ensure_btd(cap.activations))
            dy_all, bad_dy = self._sanitize_for_scoring(self._ensure_btd(cap.grad_outputs))
            topk_idx = cap.topk_idx
            topk_weight, bad_w = self._sanitize_for_scoring(cap.topk_weight)
            is_null = cap.is_null

            total_bad_act += float(bad_x)
            total_bad_grad += float(bad_dy + bad_w)

            if x_all.shape[0] < candidate_count + proxy_count:
                raise RuntimeError(
                    f"MoE layer {layer_name} captured batch too small: {x_all.shape[0]} < {candidate_count + proxy_count}"
                )

            x_cand = x_all[:candidate_count]
            dy_cand = dy_all[:candidate_count]
            idx_cand = topk_idx[:candidate_count]
            w_cand = topk_weight[:candidate_count]
            null_cand = is_null[:candidate_count]

            x_proxy = x_all[candidate_count:candidate_count + proxy_count]
            dy_proxy = dy_all[candidate_count:candidate_count + proxy_count]
            idx_proxy = topk_idx[candidate_count:candidate_count + proxy_count]
            w_proxy = topk_weight[candidate_count:candidate_count + proxy_count]
            null_proxy = is_null[candidate_count:candidate_count + proxy_count]

            feat_gate_c = torch.zeros(candidate_count, self.config.sketch_dim, device=x_all.device, dtype=out_dtype)
            feat_up_c = torch.zeros_like(feat_gate_c)
            feat_down_c = torch.zeros_like(feat_gate_c)
            feat_gate_p = torch.zeros(proxy_count, self.config.sketch_dim, device=x_all.device, dtype=out_dtype)
            feat_up_p = torch.zeros_like(feat_gate_p)
            feat_down_p = torch.zeros_like(feat_gate_p)

            wg_cache: Dict[int, torch.Tensor] = {}
            wu_cache: Dict[int, torch.Tensor] = {}
            wd_cache: Dict[int, torch.Tensor] = {}
            wg_weight_cache: Dict[int, torch.Tensor] = {}
            wu_weight_cache: Dict[int, torch.Tensor] = {}
            wd_weight_cache: Dict[int, torch.Tensor] = {}

            def _fetch_precond(cache: Dict[int, torch.Tensor], param: torch.nn.Parameter, expert_idx: int) -> torch.Tensor:
                nonlocal total_bad_precond
                if expert_idx not in cache:
                    p_raw = preconditioner.get_slice(param, expert_idx)
                    p, bad = self._sanitize_for_scoring(p_raw)
                    cache[expert_idx] = p
                    total_bad_precond += float(bad)
                return cache[expert_idx]

            def _fetch_weight(cache: Dict[int, torch.Tensor], param: torch.nn.Parameter, expert_idx: int) -> torch.Tensor:
                nonlocal total_bad_precond
                if expert_idx not in cache:
                    w_raw = param[expert_idx]
                    w, bad = self._sanitize_for_scoring(w_raw)
                    cache[expert_idx] = w
                    total_bad_precond += float(bad)
                return cache[expert_idx]

            # Bulk prefetch active expert preconditioner slices and weights once per layer.
            active_ids: List[int] = []
            if bool((~null_cand).any()):
                active_ids.extend(idx_cand[~null_cand].to(torch.long).unique().tolist())
            if bool((~null_proxy).any()):
                active_ids.extend(idx_proxy[~null_proxy].to(torch.long).unique().tolist())
            if active_ids:
                active_unique = torch.tensor(sorted(set(int(x) for x in active_ids)), device=x_all.device, dtype=torch.long)
                p_wg_all = preconditioner.get_slices(cap.w_gate, active_unique)
                p_wu_all = preconditioner.get_slices(cap.w_up, active_unique)
                p_wd_all = preconditioner.get_slices(cap.w_down, active_unique)
                for j, e_val in enumerate(active_unique.tolist()):
                    e = int(e_val)
                    p_wg, bad_wg = self._sanitize_for_scoring(p_wg_all[j])
                    p_wu, bad_wu = self._sanitize_for_scoring(p_wu_all[j])
                    p_wd, bad_wd = self._sanitize_for_scoring(p_wd_all[j])
                    wg_cache[e] = p_wg
                    wu_cache[e] = p_wu
                    wd_cache[e] = p_wd
                    total_bad_precond += float(bad_wg + bad_wu + bad_wd)
                    wg_w, bad_wgw = self._sanitize_for_scoring(cap.w_gate[e])
                    wu_w, bad_wuw = self._sanitize_for_scoring(cap.w_up[e])
                    wd_w, bad_wdw = self._sanitize_for_scoring(cap.w_down[e])
                    wg_weight_cache[e] = wg_w
                    wu_weight_cache[e] = wu_w
                    wd_weight_cache[e] = wd_w
                    total_bad_precond += float(bad_wgw + bad_wuw + bad_wdw)

            def _accumulate_partition(
                x_part: torch.Tensor,
                dy_part: torch.Tensor,
                idx_part: torch.Tensor,
                w_part: torch.Tensor,
                null_part: torch.Tensor,
                out_gate: torch.Tensor,
                out_up: torch.Tensor,
                out_down: torch.Tensor,
            ) -> None:
                nonlocal total_bad_grad
                bsz, seq_len, _ = x_part.shape
                if bsz == 0:
                    return

                # Flatten routed assignments once, then process expert groups.
                k_slots = idx_part.shape[-1]
                sample_ids_full = torch.arange(bsz, device=x_part.device, dtype=torch.long).view(bsz, 1, 1)
                sample_ids_full = sample_ids_full.expand(bsz, seq_len, k_slots)
                token_ids_full = torch.arange(seq_len, device=x_part.device, dtype=torch.long).view(1, seq_len, 1)
                token_ids_full = token_ids_full.expand(bsz, seq_len, k_slots)

                real_mask = ~null_part
                if not bool(real_mask.any()):
                    return

                sample_ids = sample_ids_full[real_mask]
                token_ids = token_ids_full[real_mask]
                expert_ids = idx_part[real_mask].to(torch.long)
                assign_w = w_part[real_mask].unsqueeze(1)

                x_tok = x_part[sample_ids, token_ids]
                dy_tok = dy_part[sample_ids, token_ids]

                # Group assignments by expert.
                order_e = expert_ids.argsort(stable=True)
                expert_ids = expert_ids[order_e]
                sample_ids = sample_ids[order_e]
                x_tok = x_tok[order_e]
                dy_tok = dy_tok[order_e]
                assign_w = assign_w[order_e]

                uniq_e, counts_e = torch.unique_consecutive(expert_ids, return_counts=True)
                starts_e = torch.cumsum(counts_e, dim=0) - counts_e

                for e_val, s_e, c_e in zip(uniq_e.tolist(), starts_e.tolist(), counts_e.tolist()):
                    e = int(e_val)
                    ee = int(s_e)
                    ff = int(s_e + c_e)

                    sids_e = sample_ids[ee:ff]
                    x_e = x_tok[ee:ff]
                    dy_e = dy_tok[ee:ff]
                    w_e = assign_w[ee:ff]

                    wg_f = _fetch_weight(wg_weight_cache, cap.w_gate, e)
                    wu_f = _fetch_weight(wu_weight_cache, cap.w_up, e)
                    wd_f = _fetch_weight(wd_weight_cache, cap.w_down, e)

                    dz_e = dy_e * w_e
                    dz_e, bad_dz = self._sanitize_for_scoring(dz_e)
                    total_bad_grad += float(bad_dz)

                    g1_e = x_e.matmul(wg_f)
                    g2_e = x_e.matmul(wu_f)
                    silu_g1_e = torch.nn.functional.silu(g1_e)
                    h_e = silu_g1_e * g2_e

                    dh_e = dz_e.matmul(wd_f.transpose(0, 1))
                    dg1_e = dh_e * g2_e * self._silu_prime(g1_e)
                    dg2_e = dh_e * silu_g1_e

                    p_wg = _fetch_precond(wg_cache, cap.w_gate, e)
                    p_wu = _fetch_precond(wu_cache, cap.w_up, e)
                    p_wd = _fetch_precond(wd_cache, cap.w_down, e)

                    # Group by sample id inside expert.
                    order_s = sids_e.argsort(stable=True)
                    sids_s = sids_e[order_s]
                    x_s = x_e[order_s]
                    h_s = h_e[order_s]
                    dz_s = dz_e[order_s]
                    dg1_s = dg1_e[order_s]
                    dg2_s = dg2_e[order_s]

                    uniq_s, counts_s = torch.unique_consecutive(sids_s, return_counts=True)
                    starts_s = torch.cumsum(counts_s, dim=0) - counts_s

                    for sid, ss, cc in zip(uniq_s.tolist(), starts_s.tolist(), counts_s.tolist()):
                        a = int(ss)
                        b = int(ss + cc)
                        sample_id = int(sid)
                        x_sel = x_s[a:b]
                        h_sel = h_s[a:b]
                        dz_sel = dz_s[a:b]
                        dg1_sel = dg1_s[a:b]
                        dg2_sel = dg2_s[a:b]

                        gate_sk, up_sk = self.projector.project_outer_sum_pair_sample(
                            left=x_sel,
                            right_a=dg1_sel,
                            right_b=dg2_sel,
                            preconditioner_a=p_wg,
                            preconditioner_b=p_wu,
                            row_dim=wg_f.shape[0],
                            col_dim=wg_f.shape[1],
                            out_dtype=out_dtype,
                            sketch_key_a=f"{layer_name}.moe.W_gate.expert{e}",
                            sketch_key_b=f"{layer_name}.moe.W_up.expert{e}",
                        )
                        down_sk = self.projector.project_outer_sum_sample(
                            left=h_sel,
                            right=dz_sel,
                            preconditioner=p_wd,
                            row_dim=wd_f.shape[0],
                            col_dim=wd_f.shape[1],
                            out_dtype=out_dtype,
                            sketch_key=f"{layer_name}.moe.W_down.expert{e}",
                        )

                        out_gate[sample_id] += gate_sk
                        out_up[sample_id] += up_sk
                        out_down[sample_id] += down_sk

            _accumulate_partition(x_cand, dy_cand, idx_cand, w_cand, null_cand, feat_gate_c, feat_up_c, feat_down_c)
            _accumulate_partition(x_proxy, dy_proxy, idx_proxy, w_proxy, null_proxy, feat_gate_p, feat_up_p, feat_down_p)

            candidate_features[f"{layer_name}.moe.W_gate"] = feat_gate_c
            candidate_features[f"{layer_name}.moe.W_up"] = feat_up_c
            candidate_features[f"{layer_name}.moe.W_down"] = feat_down_c

            proxy_features[f"{layer_name}.moe.W_gate"] = feat_gate_p.mean(dim=0)
            proxy_features[f"{layer_name}.moe.W_up"] = feat_up_p.mean(dim=0)
            proxy_features[f"{layer_name}.moe.W_down"] = feat_down_p.mean(dim=0)

            # Explicitly release per-layer local references.
            del cap, x_all, dy_all, topk_idx, topk_weight, is_null

        stats = {
            "nonfinite_moe_activations": total_bad_act,
            "nonfinite_moe_grad_outputs": total_bad_grad,
            "nonfinite_moe_preconditioner": total_bad_precond,
        }
        return candidate_features, proxy_features, stats

    def build_sketch_features(
        self,
        captures: Dict[str, LayerCapture],
        candidate_count: int,
        proxy_count: int,
        preconditioner: AdamWPreconditionerView,
        moe_captures: Optional[Dict[str, MoERoutedCapture]] = None,
        out_dtype: torch.dtype = torch.float32,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Returns:
            candidate_features: layer -> [N_local, m]
            proxy_features: layer -> [m]
        """
        candidate_features: Dict[str, torch.Tensor] = {}
        proxy_features: Dict[str, torch.Tensor] = {}

        total_bad_act = 0
        total_bad_grad = 0
        total_bad_precond = 0

        for layer_name, cap in list(captures.items()):
            a_all_raw = self._ensure_btd(cap.activations)
            g_all_raw = self._ensure_btd(cap.grad_outputs)
            a_all, bad_act = self._sanitize_for_scoring(a_all_raw)
            g_all, bad_grad = self._sanitize_for_scoring(g_all_raw)
            total_bad_act += bad_act
            total_bad_grad += bad_grad

            if a_all.shape[0] < candidate_count + proxy_count:
                raise RuntimeError(
                    f"Layer {layer_name} captured batch too small: {a_all.shape[0]} < {candidate_count + proxy_count}"
                )

            a_cand = a_all[:candidate_count]
            g_cand = g_all[:candidate_count]
            a_proxy = a_all[candidate_count:candidate_count + proxy_count]
            g_proxy = g_all[candidate_count:candidate_count + proxy_count]

            out_dim, in_dim = cap.weight.shape
            p_t_raw = preconditioner.get(cap.weight)
            p_t, bad_precond = self._sanitize_for_scoring(p_t_raw)
            total_bad_precond += bad_precond

            cand = self.projector.project_linear_batch(
                activations=a_cand,
                grad_outputs=g_cand,
                preconditioner=p_t,
                out_dim=out_dim,
                in_dim=in_dim,
                out_dtype=out_dtype,
                sketch_key=f"{layer_name}.weight",
            )

            proxy = self.projector.project_linear_batch(
                activations=a_proxy,
                grad_outputs=g_proxy,
                preconditioner=None,
                out_dim=out_dim,
                in_dim=in_dim,
                out_dtype=out_dtype,
                sketch_key=f"{layer_name}.weight",
            ).mean(dim=0)

            candidate_features[layer_name] = cand
            proxy_features[layer_name] = proxy

            # Explicitly drop raw capture buffers once sketched.
            del cap, a_all_raw, g_all_raw, a_all, g_all, a_cand, g_cand, a_proxy, g_proxy, cand, proxy

        if moe_captures:
            moe_cand, moe_proxy, moe_stats = self._build_moe_features(
                moe_captures=moe_captures,
                candidate_count=candidate_count,
                proxy_count=proxy_count,
                preconditioner=preconditioner,
                out_dtype=out_dtype,
            )
            candidate_features.update(moe_cand)
            proxy_features.update(moe_proxy)
            total_bad_act += int(moe_stats.get("nonfinite_moe_activations", 0.0))
            total_bad_grad += int(moe_stats.get("nonfinite_moe_grad_outputs", 0.0))
            total_bad_precond += int(moe_stats.get("nonfinite_moe_preconditioner", 0.0))

        if not candidate_features:
            raise RuntimeError("No candidate features were built for OPUS")
        self._last_feature_stats = {
            "nonfinite_activations": float(total_bad_act),
            "nonfinite_grad_outputs": float(total_bad_grad),
            "nonfinite_preconditioner": float(total_bad_precond),
        }
        return candidate_features, proxy_features

    @staticmethod
    def _fallback_local_random(n_local: int, k_local: int, device: torch.device, rng: torch.Generator) -> torch.Tensor:
        perm = torch.randperm(n_local, generator=rng)[:k_local]
        return perm.to(device=device, dtype=torch.long)

    def _rand_uniform_like(self, ref: torch.Tensor) -> torch.Tensor:
        flat = torch.rand(ref.numel(), generator=self._rng, dtype=torch.float32)
        return flat.view_as(ref).to(device=ref.device, dtype=ref.dtype)

    def _local_gumbel_argmax(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u = self._rand_uniform_like(logits).clamp_(min=1e-6, max=1.0 - 1e-6)
        g = -torch.log(-torch.log(u))
        perturbed = logits + g
        best_val, best_idx = torch.max(perturbed, dim=0)
        return best_val, best_idx

    @staticmethod
    def _global_pick_from_rank_bests(
        local_best_value: torch.Tensor,
        local_best_index: torch.Tensor,
        local_best_score: torch.Tensor,
        n_local: int,
        device: torch.device,
    ) -> Tuple[int, float]:
        gathered_vals = all_gather_1d(local_best_value.view(1))
        gathered_idx = all_gather_1d(local_best_index.view(1).to(torch.long))
        gathered_score = all_gather_1d(local_best_score.view(1))

        if get_rank() == 0:
            owner = int(torch.argmax(gathered_vals).item())
            best_val = float(gathered_vals[owner].item())
            if not math.isfinite(best_val) or best_val <= torch.finfo(gathered_vals.dtype).min * 0.5:
                chosen = -1
                chosen_score = 0.0
            else:
                local_idx = int(gathered_idx[owner].item())
                chosen = owner * n_local + local_idx
                chosen_score = float(gathered_score[owner].item())
        else:
            chosen = -1
            chosen_score = 0.0

        chosen_t = torch.tensor([chosen], device=device, dtype=torch.long)
        chosen_score_t = torch.tensor([chosen_score], device=device, dtype=torch.float32)
        broadcast_tensor(chosen_t, src=0)
        broadcast_tensor(chosen_score_t, src=0)
        return int(chosen_t.item()), float(chosen_score_t.item())

    def _chosen_probability(
        self,
        logits: torch.Tensor,
        chosen_score: float,
        temperature: float,
        logits_are_global: bool = False,
    ) -> float:
        # exact categorical p(chosen) from distributed logits via scalar reductions
        temp = max(float(temperature), 1e-6)
        if logits_are_global:
            global_max = logits.max().view(1).to(torch.float32)
            shifted = logits.to(torch.float32) - global_max
            denom = float(torch.exp(shifted).sum().item())
        else:
            local_max = logits.max().view(1).to(torch.float32)
            global_max = all_reduce_max(local_max.clone())
            shifted = logits.to(torch.float32) - global_max
            exp_sum_local = torch.exp(shifted).sum().view(1)
            exp_sum_global = all_reduce_sum(exp_sum_local.clone())
            denom = float(exp_sum_global.item())
        if not math.isfinite(denom) or denom <= 0.0:
            return 0.0
        num = math.exp((float(chosen_score) / temp) - float(global_max.item()))
        p = num / denom
        if not math.isfinite(p) or p <= 0.0:
            return 0.0
        return float(p)

    def select(
        self,
        candidate_features: Dict[str, torch.Tensor],
        proxy_features: Dict[str, torch.Tensor],
        learning_rate: float,
    ) -> SelectionResult:
        """
        Global Boltzmann selection with redundancy updates.
        """
        t_start = time.perf_counter()
        rank = get_rank()
        world = get_world_size()

        if not math.isfinite(float(learning_rate)):
            raise ValueError(f"learning_rate must be finite, got {learning_rate}")

        layer_names = sorted(candidate_features.keys())
        cand_raw = torch.stack([candidate_features[n] for n in layer_names], dim=1)  # [N_local, L, m]
        proxy_raw = torch.stack([proxy_features[n] for n in layer_names], dim=0)  # [L, m]
        cand, bad_cand = self._sanitize_for_scoring(cand_raw)
        proxy, bad_proxy = self._sanitize_for_scoring(proxy_raw)

        n_local = cand.size(0)
        device = cand.device

        shared_candidates = bool(self.config.zero2_exact_global_scoring) and world > 1
        if shared_candidates:
            k_global = max(1, int(round(self.config.selection_ratio * n_local)))
        else:
            k_global = max(1, int(round(self.config.selection_ratio * (n_local * world))))
        k_local_fallback = max(1, int(round(self.config.selection_ratio * n_local)))

        history = torch.zeros_like(proxy)
        selected_local = torch.zeros(n_local, dtype=torch.bool, device=device)
        selected_global_indices: List[int] = []

        alignment_acc = 0.0
        redundancy_acc = 0.0
        entropy_acc = 0.0
        nonfinite_local_scores = 0.0

        used_fallback = False
        fallback_no_finite = 0.0
        try:
            for _ in range(k_global):
                if (time.perf_counter() - t_start) > self.config.max_selector_time_s:
                    raise TimeoutError("selector timeout budget exceeded")

                alignment = torch.einsum("nlm,lm->n", cand, proxy)
                redundancy = torch.einsum("nlm,lm->n", cand, history)
                local_scores = (learning_rate * alignment) - (learning_rate * learning_rate * redundancy)
                local_scores = local_scores.to(torch.float32)
                _n_nonfinite = int((~torch.isfinite(local_scores)).sum().item())
                if self._track_nonfinite:
                    nonfinite_local_scores += float(_n_nonfinite)
                if _n_nonfinite == local_scores.numel() and local_scores.numel() > 0:
                    logger.warning(
                        "OPUS selector: all %d candidate scores are non-finite "
                        "— selection is effectively random this step",
                        local_scores.numel(),
                    )
                elif _n_nonfinite > 0:
                    logger.warning(
                        "OPUS selector: %d/%d scores are non-finite",
                        _n_nonfinite,
                        local_scores.numel(),
                    )
                local_scores = torch.nan_to_num(
                    local_scores,
                    nan=torch.finfo(torch.float32).min,
                    posinf=torch.finfo(torch.float32).min,
                    neginf=torch.finfo(torch.float32).min,
                )
                local_scores[selected_local] = -torch.inf

                if shared_candidates:
                    # Each rank contributes shard-local utility for the same candidates.
                    global_scores = all_reduce_sum(local_scores.clone())
                    temp = max(self.config.temperature, 1e-6)
                    global_logits = global_scores / temp
                    if rank == 0:
                        best_val, best_idx = self._local_gumbel_argmax(global_logits)
                        chosen = int(best_idx.item())
                        chosen_score = float(global_scores[best_idx].item())
                        if (not math.isfinite(float(best_val.item()))) or (chosen_score <= torch.finfo(torch.float32).min * 0.5):
                            chosen = -1
                            chosen_score = 0.0
                    else:
                        chosen = -1
                        chosen_score = 0.0
                    chosen_t = torch.tensor([chosen], device=device, dtype=torch.long)
                    chosen_score_t = torch.tensor([chosen_score], device=device, dtype=torch.float32)
                    broadcast_tensor(chosen_t, src=0)
                    broadcast_tensor(chosen_score_t, src=0)
                    chosen = int(chosen_t.item())
                    chosen_score = float(chosen_score_t.item())
                    local_logits = global_logits
                else:
                    temp = max(self.config.temperature, 1e-6)
                    local_logits = local_scores / temp
                    local_best_val, local_best_idx = self._local_gumbel_argmax(local_logits)
                    local_best_score = local_scores[local_best_idx]
                    chosen, chosen_score = self._global_pick_from_rank_bests(
                        local_best_value=local_best_val.to(torch.float32),
                        local_best_index=local_best_idx.to(torch.long),
                        local_best_score=local_best_score.to(torch.float32),
                        n_local=n_local,
                        device=device,
                    )
                if chosen < 0:
                    fallback_no_finite = 1.0
                    break

                selected_global_indices.append(chosen)
                if shared_candidates:
                    local_idx = chosen
                    selected_local[local_idx] = True
                    selected_feat = cand[local_idx].clone()
                else:
                    owner = chosen // n_local
                    local_idx = chosen % n_local
                    if rank == owner:
                        selected_local[local_idx] = True
                        selected_feat = cand[local_idx].clone()
                    else:
                        selected_feat = torch.zeros_like(history)

                p = self._chosen_probability(
                    local_logits,
                    chosen_score,
                    self.config.temperature,
                    logits_are_global=shared_candidates,
                )
                if p > 0.0:
                    entropy_acc += float(-p * math.log(p))
                alignment_acc += float(chosen_score)
                work = all_reduce_sum_async(selected_feat)
                if work is not None:
                    work.wait()
                else:
                    all_reduce_sum(selected_feat)
                history = history + selected_feat
                redundancy_acc += float(torch.norm(selected_feat).item())

        except Exception:
            if not self.config.fallback_random_on_error:
                raise
            used_fallback = True
            if shared_candidates:
                if rank == 0:
                    local_idxs = self._fallback_local_random(n_local, k_local_fallback, device, self._rng)
                else:
                    local_idxs = torch.empty(k_local_fallback, device=device, dtype=torch.long)
                broadcast_tensor(local_idxs, src=0)
                global_idxs = local_idxs.clone()
            else:
                local_idxs = self._fallback_local_random(n_local, k_local_fallback, device, self._rng)
                global_idxs = local_idxs + (rank * n_local)
            return SelectionResult(
                selected_local_indices=local_idxs,
                selected_global_indices=global_idxs,
                used_fallback=True,
                metrics={
                    "alignment": 0.0,
                    "redundancy": 0.0,
                    "entropy": 0.0,
                    "nonfinite_feature_values": float(
                        bad_cand + bad_proxy + self._last_feature_stats.get("nonfinite_activations", 0.0)
                        + self._last_feature_stats.get("nonfinite_grad_outputs", 0.0)
                        + self._last_feature_stats.get("nonfinite_preconditioner", 0.0)
                    ),
                    "nonfinite_local_score_values": nonfinite_local_scores,
                    "fallback_no_finite_scores": fallback_no_finite,
                    "selector_time_s": float(time.perf_counter() - t_start),
                },
            )

        local_indices = torch.nonzero(selected_local, as_tuple=False).flatten()
        if local_indices.numel() == 0:
            # Hard floor to avoid empty update on any rank.
            local_indices = self._fallback_local_random(n_local, k_local_fallback, device, self._rng)
            used_fallback = True
            fallback_no_finite = 1.0

        if selected_global_indices:
            global_idx_tensor = torch.tensor(selected_global_indices, device=device, dtype=torch.long)
        else:
            if shared_candidates:
                global_idx_tensor = local_indices.clone()
            else:
                global_idx_tensor = local_indices + (rank * n_local)

        return SelectionResult(
            selected_local_indices=local_indices,
            selected_global_indices=global_idx_tensor,
            used_fallback=used_fallback,
            metrics={
                "alignment": alignment_acc,
                "redundancy": redundancy_acc,
                "entropy": entropy_acc,
                "nonfinite_feature_values": float(
                    bad_cand + bad_proxy + self._last_feature_stats.get("nonfinite_activations", 0.0)
                    + self._last_feature_stats.get("nonfinite_grad_outputs", 0.0)
                    + self._last_feature_stats.get("nonfinite_preconditioner", 0.0)
                ),
                "nonfinite_local_score_values": nonfinite_local_scores,
                "fallback_no_finite_scores": fallback_no_finite,
                "selector_time_s": float(time.perf_counter() - t_start),
            },
        )
