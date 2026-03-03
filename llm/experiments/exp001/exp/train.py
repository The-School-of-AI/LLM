from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, ContextManager

import deepspeed
import torch
import yaml
from exp.distributed import all_reduce_sum, get_rank, set_seed
from exp.opus import (
    AdamWPreconditionerView,
    CountSketchProjector,
    OpusGhostCollector,
    OpusSelector,
    RandomInDistributionProxyProvider,
    SelectionResult,
)
from exp.proxy_dataset import ProxyDatasetConfig, get_proxy_dataloader
from transformers.tokenization_utils_tokenizers import TokenizersBackend

from llm.data import get_dataloaders, get_tokenizer
from llm.kernels.triton_cross_entropy import (
    FusedLinearCrossEntropyLoss as FusedLinearCE,
)
from llm.models import KroneckerConfig, KroneckerEmbeddings, Model1B, ModelConfig
from llm.profiler import PipelineProfiler, StepProfiler
from llm.utils import print_rank_0

# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    max_length: int = 128
    dataset_name: str | None = None
    dataset_config: str | None = None
    tokenized_dataset_path: str | None = None
    dataset_cache_dir: str | None = None
    local_nvme_cache_dir: str | None = None
    require_local_nvme: bool = False
    pack_into_blocks: bool = False
    block_sizes: list[int] | None = None
    block_size_counts: dict[Any, Any] | None = None
    domain_column: str | None = None
    concat_across_domains: bool = False
    drop_remainder: bool = True
    num_workers: int = 12
    tokenize_num_proc: int | None = None


@dataclass
class TrainConfig:
    max_steps: int
    log_interval: int = 10
    profile_steps: list[int] | None = None


@dataclass
class OpusConfig:
    # How many times larger the candidate pool is vs. the training batch size.
    # e.g. candidate_multiplier=4 with train_micro_batch_size_per_gpu=8 means
    # each GPU draws 32 candidates and selects the best ~16 of them.
    candidate_multiplier: int

    # Number of proxy samples to draw from the proxy pool each step to estimate
    # the target gradient direction g_proxy.
    n_proxy: int

    # Dimensionality of the CountSketch projection space. Higher = more accurate
    # inner-product estimates but more memory. 8192 is a good default.
    sketch_dim: int

    # Boltzmann sampling temperature τ. Higher = more uniform sampling (more
    # diversity), lower = greedier (closer to top-K).
    temperature: float

    # Seed for the CountSketch hash functions. Must be identical across all GPUs
    # so every GPU builds the same projection matrices.
    sketch_seed: int

    # If True, treat parameters whose optimizer state is not local to this GPU
    # shard (ZeRO-3) as having zero preconditioner rather than falling back to
    # a scalar approximation.
    strict_shard_preconditioner: bool = True

    # Maximum wall-clock time in seconds allowed for one full selection loop.
    # If exceeded, the selector raises TimeoutError and (if fallback_random_on_error
    # is True) falls back to random selection for that step.
    max_selector_time_s: float = 30.0

    # If True, any exception in the selection loop (including timeout) causes a
    # silent fallback to random candidate selection rather than crashing the run.
    # Set to False during debugging to surface errors immediately.
    fallback_random_on_error: bool = True


@dataclass
class Config:
    seed: int
    deepspeed_config: str
    tokenizer_dir: str
    profiler_output_dir: str
    data: DataConfig
    proxy: ProxyDatasetConfig
    train: TrainConfig
    opus: OpusConfig
    model: ModelConfig


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    def __init__(self, local_rank: int, c: Config):
        with open(c.deepspeed_config, "r") as f:
            ds_config = yaml.safe_load(f)

        # Override the scheduler's total_num_steps from TrainConfig so there is
        # a single source of truth for max_steps — no need to keep config.yaml
        # and deepspeed_config.yaml in sync manually.
        if "scheduler" in ds_config and "params" in ds_config["scheduler"]:
            ds_config["scheduler"]["params"]["total_num_steps"] = c.train.max_steps

        set_seed(c.seed)
        self.config = c
        self.pipe_prof = PipelineProfiler(
            rank=local_rank, output_dir=c.profiler_output_dir
        )

        # ── Tokenizer ────────────────────────────────────────────────────────
        with self.pipe_prof.stage("tokenizer_load"):
            print_rank_0("loading tokenizer")
            tokenizer = get_tokenizer(c.tokenizer_dir)

        # ── Dataloaders ──────────────────────────────────────────────────────
        with self.pipe_prof.stage("data_load"):
            print_rank_0("loading dataloaders")

            # DeepSpeed's train_micro_batch_size_per_gpu is the number of
            # sequences the optimizer step actually trains on per GPU. OPUS
            # draws candidate_multiplier times that many sequences and selects
            # the best train_micro_batch_size_per_gpu of them.
            #
            # selection_ratio = 1 / candidate_multiplier, which means we always
            # select back exactly train_micro_batch_size_per_gpu samples on
            # average — keeping DeepSpeed's gradient accounting consistent.
            selected_batch_size = ds_config["train_micro_batch_size_per_gpu"]
            candidate_pool_size = selected_batch_size * c.opus.candidate_multiplier
            selection_ratio = 1.0 / c.opus.candidate_multiplier
            initial_lr = float(ds_config["optimizer"]["params"]["lr"])

            self.train_loader, _, _, _ = get_dataloaders(
                tokenizer=tokenizer,
                batch_size=candidate_pool_size,
                **asdict(c.data),
            )

            proxy_loader = get_proxy_dataloader(
                tokenizer=tokenizer,
                config=c.proxy,
                seed=c.seed,
            )
            self.proxy_provider = RandomInDistributionProxyProvider(proxy_loader)

        # ── Model ────────────────────────────────────────────────────────────
        with self.pipe_prof.stage("kronecker_vocab_build"):
            print_rank_0("building kronecker embeddings")
            bpe_vocab, k_embed = Trainer._build_kronecker_vocab(tokenizer)

        with self.pipe_prof.stage("model_build"):
            print_rank_0("building model")
            model = Model1B(
                config=c.model,
                embedding_type="kronecker",
                bpe_vocab=bpe_vocab,
                pf_codec=k_embed,
            )

        with self.pipe_prof.stage("model_to_bf16"):
            print_rank_0("casting model to bfloat16")
            model = model.to(dtype=torch.bfloat16)

        # ── DeepSpeed ────────────────────────────────────────────────────────
        self.engine: deepspeed.DeepSpeedEngine
        with self.pipe_prof.stage("deepspeed_init"):
            self.engine, self.optimizer, self.lr_scheduler, _ = deepspeed.initialize(
                config_params=ds_config, model=model
            )

        print_rank_0(f"ZeRO Stage: {self.engine.zero_optimization_stage()}")

        self.step_prof: StepProfiler | None = None
        if c.train.profile_steps:
            self.step_prof = StepProfiler(
                rank=local_rank,
                profile_steps=set(c.train.profile_steps),
                output_dir=c.profiler_output_dir,
            )
            self.step_prof.activate()
            self.step_prof.register_model(self.engine.module)

        # DeepSpeed always sets engine.device after initialize() — assert here
        # so the type-checker knows it's not None for the rest of the code.
        assert self.engine.device is not None, "DeepSpeed engine has no device"
        self.device: torch.device = self.engine.device

        # ── OPUS components ──────────────────────────────────────────────────

        # AdamW preconditioner: reads optimizer state (v_hat) to build P_t.
        # We call .refresh() at the start of each step to snapshot the latest state.
        self.preconditioner_view = AdamWPreconditionerView(
            self.optimizer,
            strict_shard_only=c.opus.strict_shard_preconditioner,
        )

        # FusedLinearCE: shared instance for both scoring and training passes.
        # Fuses lm_head matmul + cross-entropy to avoid materialising [B*T, vocab].
        self._fused_ce = FusedLinearCE(ignore_index=-100, reduction="mean")

        # CountSketch projector: projects high-dimensional preconditioned
        # gradients into a sketch_dim-dimensional space for cheap inner products.
        # Same seed on all GPUs → identical hash maps → comparable sketches.
        self.sketcher = CountSketchProjector(
            sketch_dim=c.opus.sketch_dim,
            seed=c.opus.sketch_seed,
        )

        # OpusSelector: runs the iterative Boltzmann selection loop after the
        # ghost collector has produced alignment scores and candidate sketches.
        # selection_ratio is derived from candidate_multiplier so the expected
        # number of selected samples per GPU always equals selected_batch_size.
        # learning_rate is no longer stored at construction — it is passed
        # explicitly to select() each step so it tracks the LR schedule.
        self.selector = OpusSelector(
            selection_ratio=selection_ratio,
            temperature=c.opus.temperature,
            seed=c.seed,
            max_selector_time_s=c.opus.max_selector_time_s,
            fallback_random_on_error=c.opus.fallback_random_on_error,
        )

        # Store for use in the train loop
        self.selected_batch_size = selected_batch_size
        self.candidate_pool_size = candidate_pool_size
        self.selection_ratio = selection_ratio
        self.initial_lr = initial_lr

    def train(self):
        c = self.config
        device = self.device
        global_step = 0

        print_rank_0("starting training")

        for step, batch in enumerate(self.train_loader):
            if self.step_prof is not None:
                self.step_prof.start_step(global_step=step)

            # ── Move candidate pool to device ─────────────────────────────
            # The train loader yields candidate_pool_size sequences per GPU.
            # Shape: (candidate_pool_size, full_seq_len)
            candidate_ids: torch.Tensor = batch["input_ids"].to(
                device, non_blocking=True
            )
            n_candidates = candidate_ids.size(0)

            # ── OPUS selection ───────────────────────────────────────────
            # When candidate_multiplier == 1 there are no extra candidates to
            # choose from, so skip the entire scoring pass and use all samples
            # directly. This acts as a clean baseline (random / no selection).
            if c.opus.candidate_multiplier == 1:
                local_indices = torch.arange(n_candidates, device=device)
                # Reconstruct global indices using the same encoding as the
                # selector: global_idx = owner_rank * n_local + local_idx.
                # Works correctly under ZeRO-0/2/3 and single-GPU alike.
                rank_offset = get_rank() * n_candidates
                global_indices = local_indices + rank_offset
                result = SelectionResult(
                    selected_local_indices=local_indices,
                    selected_global_indices=global_indices,
                    used_fallback=True,
                    metrics={},
                )
                current_lr = self.initial_lr
            else:
                # Runs proxy sampling, scoring pass (forward+backward with ghost
                # hooks), Boltzmann selection, and index extraction.
                # See _select_candidates() for the full breakdown.
                with self._step_phase("step/opus_selection"):
                    local_indices, result, current_lr = self._select_candidates(
                        candidate_ids
                    )

            # ── Training pass (Pass 2) ────────────────────────────────────
            # Uses the full-length candidate sequences (not the proxy-length
            # truncated ones used for scoring) so training sees the complete
            # context window.
            # Shape: (k_local, full_seq_len)
            selected_ids = candidate_ids[local_indices]

            # Forward — identical loss formulation to the scoring pass so the
            # gradient signal is consistent throughout.
            with self._step_phase("step/train_forward"):
                train_loss = self._forward_and_loss(selected_ids)

            # Backward — the only backward that counts toward a weight update.
            with self._step_phase("step/train_backward"):
                self.engine.backward(train_loss)

            # Optimizer step — updates weights using the accumulated gradients.
            with self._step_phase("step/train_optimizer_step"):
                self.engine.step()

            # ── Logging ───────────────────────────────────────────────────
            global_step += 1
            loss_val = train_loss.detach().float().item()

            if self.step_prof is not None:
                _ptoks = torch.tensor(
                    selected_ids.numel(), dtype=torch.long, device=self.device
                )
                _ptoks = all_reduce_sum(_ptoks)
                self.step_prof.end_step(tokens=int(_ptoks.item()))

            if global_step % c.train.log_interval == 0:
                n_selected_local = local_indices.numel()
                m = result.metrics
                print_rank_0(
                    f"step {global_step}/{c.train.max_steps} | "
                    f"loss {loss_val:.4f} | "
                    f"lr {current_lr:.2e} | "
                    f"selected {n_selected_local}/{n_candidates} local samples | "
                    f"alignment {m.get('alignment', 0.0):.3f} | "
                    f"redundancy {m.get('redundancy', 0.0):.3f} | "
                    f"entropy {m.get('entropy', 0.0):.3f} | "
                    f"selector_time {m.get('selector_time_s', 0.0) * 1000:.1f}ms"
                    + (" | [FALLBACK]" if result.used_fallback else "")
                )

            # ── 9. Stop at max_steps ────────────────────────────────────────
            if global_step >= c.train.max_steps:
                print_rank_0(f"reached max_steps={c.train.max_steps}, stopping.")
                break

        print_rank_0("training complete.")

    def write_reports(self):
        self.pipe_prof.write_report()
        self.pipe_prof.write_jsonl()
        if self.step_prof is not None:
            self.step_prof.write_report()
            self.step_prof.write_jsonl()

    def _select_candidates(
        self,
        candidate_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, SelectionResult, float]:
        """
        Run the full OPUS selection pipeline for one training step.

        This covers everything between receiving the candidate pool and producing
        the final local indices for the training pass:
            1. Sample proxy sequences
            2. Scoring pass — forward+backward over [proxy | candidates] with
               ghost hooks to collect alignment scores and candidate sketches
            3. Boltzmann selection — distributed iterative loop
            4. Extract local indices from the selection result

        Args:
            candidate_ids: (candidate_pool_size, full_seq_len) token ids for
                           this GPU's local candidate pool.

        Returns:
            local_indices:  (k_local,)        indices into candidate_ids for training
            result:         SelectionResult    metrics + global index info
            current_lr:     float              LR at this step, for logging + scoring
        """
        c = self.config
        device = self.device
        n_candidates = candidate_ids.size(0)

        # ── snapshot the AdamW preconditioner state ───────────────────
        # Done before any gradients are computed so P_t reflects the
        # optimizer's state at the start of this step (v_{t-1}).
        with self._step_phase("opus/preconditioner_refresh"):
            self.preconditioner_view.refresh()

        # ── Sample proxy sequences ────────────────────────────────────────────
        # Draw n_proxy sequences from the in-distribution proxy pool. These
        # estimate the target gradient direction g_proxy for this step.
        # seq_len is fixed at proxy_seq_len (e.g. 512) — candidates will be
        # truncated to match before the scoring pass.
        with self._step_phase("opus/proxy_sample"):
            proxy_ids: torch.Tensor = self.proxy_provider.sample(
                device=device,  # type: ignore[arg-type]  # asserted non-None in __init__
                k=c.opus.n_proxy,
                seq_len=c.proxy.seq_len,
            )
        n_proxy = proxy_ids.size(0)
        proxy_seq_len = proxy_ids.size(1)

        # ── Build combined scoring batch ──────────────────────────────────────
        # Truncate candidates to proxy_seq_len so proxy and candidate sketches
        # live in comparable feature spaces. Full-length candidates are kept
        # separately and used untouched in the training pass.
        scoring_candidate_ids = candidate_ids[:, :proxy_seq_len]
        # Shape: (n_proxy + n_candidates, proxy_seq_len)
        combined_ids = torch.cat([proxy_ids, scoring_candidate_ids], dim=0)

        # ── Scoring pass: forward ─────────────────────────────────────────────
        # A single forward pass over [proxy | candidates] fires the ghost hooks
        # in every linear layer. The hooks compute per-layer sketches on the fly
        # without ever materialising the full [out_dim, in_dim] gradient matrix.
        ghost_collector = OpusGhostCollector(
            model=self.engine.module,
            n_proxy=n_proxy,
            n_candidates=n_candidates,
            preconditioner=self.preconditioner_view,
            sketcher=self.sketcher,
            device=device,  # type: ignore[arg-type]  # asserted non-None in __init__
        )

        with ghost_collector:
            with self._step_phase("opus/scoring_forward"):
                scoring_loss = self._forward_and_loss(combined_ids)

            # Backward fires the ghost hooks. We deliberately do NOT call
            # engine.step() — this pass is for scoring only.
            with self._step_phase("opus/scoring_backward"):
                self.engine.backward(scoring_loss)

            # Discard scoring gradients so they don't pollute the training pass.
            with self._step_phase("opus/zero_grad"):
                self.engine.zero_grad()

            alignment_scores, candidate_sketches = ghost_collector.results()

        # ── Boltzmann selection ───────────────────────────────────────────────
        # OpusSelector runs the distributed Boltzmann loop:
        #   - Each GPU nominates its local Gumbel-argmax winner
        #   - Winners compete globally via _global_pick_from_rank_bests
        #   - History Φ is updated on all GPUs via all_reduce after each pick
        # Returns selected_local_indices directly — no manual remapping needed.
        current_lr = (
            self.engine.get_lr()[0]
            if hasattr(self.engine, "get_lr")
            else self.initial_lr
        )
        with self._step_phase("opus/boltzmann_select"):
            result = self.selector.select(
                alignment_scores=alignment_scores,
                candidate_sketches=candidate_sketches,
                learning_rate=current_lr,
            )

        return result.selected_local_indices, result, current_lr

    # -------------------------------------------------------------------------

    def _forward_and_loss(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass and compute loss exactly as the reference train.py does.

        Matches the custom recurrence forward signature of Model1B:
          - Shifts tokens manually: x_input = ids[:, :-2], y_ntp = ids[:, 1:-1]
          - Calls engine with return_hidden=True so lm_head is skipped
          - next_token_ids=None disables MTP (unwanted for OPUS scoring)
          - Computes NTP loss via FusedLinearCE (fused lm_head matmul + CE)
          - Adds auxiliary loss (GSA/routing) if present

        Using the same loss formulation in both the scoring pass and training pass
        is critical for OPUS: the ghost hooks capture gradients of this exact loss,
        so the utility scores estimate alignment with the same objective that the
        optimizer actually minimises.

        Args:
            input_ids: (B, T) token ids. Must have T >= 3 (two tokens are consumed
                       by the NTP shift, one is dropped from the end).

        Returns:
            Scalar loss tensor with grad_fn attached.
        """
        # Token shifting — matches reference train.py exactly:
        #   x_input : ids[:, :-2]  fed into the model
        #   y_ntp   : ids[:, 1:-1] NTP targets (one step ahead of x_input)
        # The last token of x_input predicts y_ntp[-1], and the second-to-last
        # token of input_ids is never used as a target — same as the reference.
        x_input = input_ids[:, :-2].contiguous()
        y_ntp = input_ids[:, 1:-1].contiguous()

        # Custom forward: return_hidden=True skips lm_head so FusedLinearCE can
        # fuse the matmul + cross-entropy without materialising [B*T, vocab].
        # next_token_ids=None disables MTP unconditionally.
        h_ntp, _h_mtp, aux_loss = self.engine(
            x_input,
            next_token_ids=None,
            attention_mask=None,
            return_loss=True,
            return_memory=False,
            prev_memory_stream=None,
            return_hidden=True,
        )

        # FusedLinearCE: fuses lm_head matmul + cross-entropy in one Triton kernel.
        # Avoids materialising the full [B*T, vocab_size] logit tensor.
        lm_weight = self.engine.module.lm_head.weight  # (vocab_size, hidden_size)
        B, T, H = h_ntp.shape
        loss = self._fused_ce(
            h_ntp.view(-1, H),  # (B*T, H)
            lm_weight,  # (vocab_size, H)
            y_ntp.view(-1),  # (B*T,)
        )

        # Add auxiliary loss (GSA sparse attention routing etc.) if present
        if aux_loss is not None and aux_loss.numel() > 0:
            aux_term = aux_loss if aux_loss.numel() == 1 else aux_loss.mean()
            loss = loss + aux_term

        return loss

    def _step_phase(self, phase: str) -> ContextManager:
        return self.step_prof.phase(phase) if self.step_prof else _null_ctx()

    @staticmethod
    def _build_kronecker_vocab(
        tokenizer: TokenizersBackend,
    ) -> tuple[list[str], KroneckerEmbeddings]:
        # Use len(tokenizer) to include special tokens (pad, eos, etc.)
        vocab_size = len(tokenizer)
        bpe_vocab = []
        for i in range(vocab_size):
            try:
                token = tokenizer.decode([i])
                bpe_vocab.append(token if token else f"<unk_{i}>")
            except Exception:
                bpe_vocab.append(f"<unk_{i}>")

        # Create Kronecker codec
        pf_config = KroneckerConfig(
            CHAR_DIM=256,
            POS_DIM=32,
            D=8192,
            length_normalize=True,
            truncate_long_words=True,
        )

        return bpe_vocab, KroneckerEmbeddings(pf_config)


@contextmanager
def _null_ctx():
    yield
