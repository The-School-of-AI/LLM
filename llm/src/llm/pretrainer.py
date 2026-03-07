import time

import torch
import torch.distributed as dist
import yaml
from torch import Tensor
from tqdm import tqdm

import llm.factories as ft
from llm.config import Config
from llm.deepspeed_config import apply_runtime_overrides
from llm.kernels import HAS_TRITON, FusedLinearCrossEntropyLoss
from llm.logger import Metrics
from llm.profiler import PipelineProfiler
from llm.utils import is_main_process


class PreTrainer:
    def __init__(self, local_rank: int, c: Config):
        self._config = c
        self.run_id: str = c.run_id  # type: ignore Config.__post_init__ ensures run id is not None
        self._validate_kernel_policy(c.training.require_fused_kernels)

        with open(c.training.deepspeed_config, "r") as f:
            ds_config = yaml.safe_load(f)
        ds_config = apply_runtime_overrides(
            ds_config, c.training.overlap_communication
        )
        batch_size_per_gpu = ds_config["train_micro_batch_size_per_gpu"]

        self._pipe_prof = PipelineProfiler(rank=local_rank)

        with self._pipe_prof.stage("tokenizer_load"):
            tokenizer = ft.build_tokenizer(c.data)

        with self._pipe_prof.stage("data_load"):
            (
                self._train_loader,
                self._train_sampler,
                self._val_loader,
                self._val_sampler,
                self._test_loader,
                self._test_sampler,
            ) = ft.build_dataloaders(c.data, batch_size_per_gpu, tokenizer)

        with self._pipe_prof.stage("model_build"):
            model = ft.build_model(c.model, tokenizer)

        self._fused_ce_fn = FusedLinearCrossEntropyLoss(
            ignore_index=-100,
            reduction="mean",
            max_chunk_gb=c.training.fused_ce_chunk_gb,
        )

        with self._pipe_prof.stage("deepspeed_init"):
            self._engine = ft.build_deepspeed(model, ds_config)
            self._optimizer = self._engine.optimizer
            self._lr_scheduler = self._engine.lr_scheduler

        self._ckpt_manager = ft.build_checkpoint_manager(
            c.checkpoint, c.checkpoints_dir
        )

        self._logger = ft.build_observability(c.observability, self.run_id, local_rank)
        self._step_profiler = ft.build_step_profiler(c.training, local_rank)
        self._step_profiler.activate()
        self._step_profiler.register_model(self._engine.module)

    def run(self):
        start_epoch, start_step, global_step = self._resume()
        max_epochs = self._config.training.max_epochs
        max_steps_per_epoch = self._config.training.max_steps_per_epoch
        device = self._engine.device
        ckpt_interval = self._config.checkpoint.save_interval

        for epoch in range(start_epoch, max_epochs):
            if self._train_sampler:
                self._train_sampler.set_epoch(epoch)

            progress_bar = tqdm(
                self._train_loader, desc=f"epoch {epoch}", disable=not is_main_process()
            )

            steps = 0
            total_loss = 0.0
            self._engine.train()
            for step, batch in enumerate(progress_bar):
                if step < start_step:
                    continue

                self._step_profiler.start_step(global_step)
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                if (
                    torch.cuda.is_available()
                ):  # TODO: do this only if profiler is enabled.
                    torch.cuda.synchronize()

                metrics = Metrics()

                step_start_time = time.time()

                with self._step_profiler.phase("step/train_forward"):
                    loss, forward_metrics = self._forward(
                        epoch, step, input_ids, attention_mask
                    )
                with self._step_profiler.phase("step/train_backward"):
                    self._backward(loss)
                with self._step_profiler.phase("step/train_optimizer_step"):
                    self._optimizer_step()
                self._step_profiler.end_step()
                step_time = time.time() - step_start_time
                num_tokens = torch.tensor(
                    input_ids.numel(), dtype=torch.float32, device=device
                )
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM)
                self._step_profiler.end_step(tokens=int(num_tokens.item()))
                toks_per_sec = num_tokens.item() / step_time if step_time > 0 else 0

                if ckpt_interval is not None and (step + 1) % ckpt_interval == 0:
                    self._save_checkpoint(
                        epoch,
                        step,
                        global_step,
                        loss=loss.item(),
                    )

                global_step += 1
                steps += 1
                total_loss += loss.item()

                metrics.add("loss", loss.item(), pbar=True)
                metrics.add("global_step", global_step, pbar=True)
                metrics.add("toks/sec", toks_per_sec, pbar=True)
                metrics = metrics | forward_metrics

                progress_bar.set_postfix(metrics.get_pbar_values())
                self._logger.log_metrics(global_step, metrics)

                if max_steps_per_epoch is not None and step >= max_steps_per_epoch:
                    break

            start_step = 0
            avg_loss = total_loss / steps if steps > 0 else 0

            with self._pipe_prof.stage(f"epoch_{epoch}_val"):
                val_loss, val_perplexity = self._validate()

            self._save_checkpoint(
                epoch + 1,  # epoch to resume from
                0,
                global_step,
                epoch_end=True,
                avg_loss=avg_loss,
                val_loss=val_loss,
                val_perplexity=val_perplexity,
            )

        self._cleanup()

    def _resume(self) -> tuple[int, int, int]:
        ckpt_config = self._config.checkpoint
        if not ckpt_config.resume_from:
            return 0, 0, 0

        if self._ckpt_manager is None:
            raise RuntimeError(
                "cannot resume from checkpoint when no checkpoint manager is configured"
            )

        client_state = self._ckpt_manager.load_checkpoint(
            self._engine,
            step=ckpt_config.resume_step or 0,
            tag=ckpt_config.resume_from,
        )

        if client_state:
            start_epoch = client_state.get("epoch", 0)
            global_step = client_state.get("global_step", 0)
            start_step = client_state.get("step", 0)
            return start_epoch, start_step, global_step

        return 0, 0, 0

    def _forward(
        self,
        epoch: int,
        step: int,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Metrics]:
        model_variant = self._config.model.variant
        loss: Tensor
        metrics = Metrics()

        match model_variant:
            case "1b_reversible":
                loss, _metrics = self._1b_forward(
                    epoch, step, input_ids, attention_mask
                )
                metrics = metrics | _metrics

            case _:
                raise RuntimeError(
                    f"cannot train on an unknown model variant {model_variant}"
                )

        return loss, metrics

    def _1b_forward(
        self,
        epoch: int,
        step: int,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Metrics]:
        metrics = Metrics()

        # Reversible model: returns (h_ntp, h_mtp, aux_loss) hidden states
        # (NOT logits — lm_head is skipped; FusedLinearCE fuses matmul+CE below)
        x_input = input_ids[:, :-2].contiguous()
        y_ntp = input_ids[:, 1:-1].contiguous()
        y_mtp = input_ids[:, 2:].contiguous()

        h_ntp, h_mtp, aux_loss = self._engine(
            x_input,
            next_token_ids=y_ntp,
            attention_mask=(
                attention_mask[:, :-2].contiguous()
                if attention_mask is not None
                else None
            ),
            return_loss=True,
            return_memory=False,
            prev_memory_stream=None,
            return_hidden=True,  # Skip lm_head — we compute CE below
        )

        if self._config.training.check_for_gsa_leak:
            gsa_leak_frac = None
            gsa_leak_attempt_frac = None
            leak_frac_t = getattr(self._engine.module, "last_gsa_leak_fraction", None)
            leak_attempt_t = getattr(
                self._engine.module, "last_gsa_leak_attempt_fraction", None
            )
            if leak_frac_t is not None:
                leak_frac_t = leak_frac_t.detach().float()
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(leak_frac_t, op=dist.ReduceOp.SUM)
                    leak_frac_t = leak_frac_t / dist.get_world_size()
                gsa_leak_frac = float(leak_frac_t.item())
            if leak_attempt_t is not None:
                leak_attempt_t = leak_attempt_t.detach().float()
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(leak_attempt_t, op=dist.ReduceOp.SUM)
                    leak_attempt_t = leak_attempt_t / dist.get_world_size()
                gsa_leak_attempt_frac = float(leak_attempt_t.item())

            if gsa_leak_frac is not None and gsa_leak_frac > 1e-12:
                raise RuntimeError(
                    f"GSA causal leak regression detected: gsa_leak_fraction={gsa_leak_frac:.6e}"
                )
            metrics.add("gsa_leak_attempt_fraction", gsa_leak_attempt_frac, pbar=True)

        _, _, H_dim = h_ntp.shape
        lm_weight = self._engine.module.lm_head.weight
        loss_ntp = self._fused_ce_fn(
            h_ntp.view(-1, H_dim),  # [B*T, H]
            lm_weight,  # [V, H]
            y_ntp.view(-1),  # [B*T]
        )

        loss_mtp = None
        if h_mtp is not None:
            _, _, H_m = h_mtp.shape
            loss_mtp = self._fused_ce_fn(
                h_mtp.view(-1, H_m),  # [B*T, H]
                lm_weight,  # [V, H]
                y_mtp.view(-1),  # [B*T]
            )

        if (
            torch.isnan(loss_ntp)
            or (loss_mtp is not None and torch.isnan(loss_mtp))
            or (aux_loss is not None and torch.isnan(aux_loss))
        ):
            raise RuntimeError(
                f"NaN detected at epoch {epoch}, step {step}: "
                f"loss_ntp={loss_ntp.item():.4f}, "
                f"loss_mtp={loss_mtp.item():.4f if loss_mtp is not None else 'None'}"  # type: ignore
            )

        loss = loss_ntp
        if loss_mtp is not None:
            loss = loss + 0.3 * loss_mtp
        if aux_loss is not None and aux_loss.numel() > 0:
            # Defensive scalarization: some model variants may return
            # aux tensors with more than one element.
            aux_term = aux_loss if aux_loss.numel() == 1 else aux_loss.mean()
            loss = loss + aux_term

        metrics.add("loss_ntp", float(loss_ntp.detach().float().item()), pbar=True)
        metrics.add(
            "loss_mtp",
            (float(loss_mtp.detach().float().item()) if loss_mtp is not None else None),
            pbar=True,
        )
        metrics.add(
            "loss_aux",
            (float(aux_loss.detach().float().item()) if aux_loss is not None else None),
            pbar=True,
        )

        return loss, metrics

    def _backward(self, loss: Tensor):
        self._engine.backward(loss)

    def _optimizer_step(self):
        self._engine.step()

    def _validate(self) -> tuple[float, float]:
        self._engine.eval()
        total_loss = 0.0
        total_perplexity = 0.0
        steps = 0
        max_steps = self._config.training.max_val_steps

        lm_weight = self._engine.module.lm_head.weight  # [V, H]

        progress_bar = tqdm(
            self._val_loader, desc="validation", disable=not is_main_process()
        )

        with torch.no_grad():
            for i, batch in enumerate(progress_bar):
                input_ids = batch["input_ids"].to(
                    self._engine.device, non_blocking=True
                )
                attention_mask = batch["attention_mask"].to(
                    self._engine.device, non_blocking=True
                )

                # Mirror the training path: use return_hidden=True so the model
                # returns hidden states instead of [B, T, vocab] logits. This
                # avoids materialising the enormous logit tensor and lets us
                # reuse FusedLinearCE (chunked matmul + CE) for memory safety.
                x_input = input_ids[:, :-2].contiguous()
                y_ntp = input_ids[:, 1:-1].contiguous()
                y_mtp = input_ids[:, 2:].contiguous()

                h_ntp, h_mtp, aux_loss = self._engine.module(
                    x_input,
                    next_token_ids=y_ntp,
                    attention_mask=(
                        attention_mask[:, :-2].contiguous()
                        if attention_mask is not None
                        else None
                    ),
                    return_loss=True,
                    return_memory=False,
                    prev_memory_stream=None,
                    return_hidden=True,  # Skip lm_head — FusedLinearCE handles it
                )

                _, _, H_dim = h_ntp.shape
                loss_ntp = self._fused_ce_fn(
                    h_ntp.view(-1, H_dim),  # [B*T, H]
                    lm_weight,  # [V, H]
                    y_ntp.view(-1),  # [B*T]
                )

                loss_mtp = None
                if h_mtp is not None:
                    _, _, H_m = h_mtp.shape
                    loss_mtp = self._fused_ce_fn(
                        h_mtp.view(-1, H_m),  # [B*T, H]
                        lm_weight,  # [V, H]
                        y_mtp.view(-1),  # [B*T]
                    )

                loss = loss_ntp
                if loss_mtp is not None:
                    loss = loss + 0.3 * loss_mtp
                if aux_loss is not None and aux_loss.numel() > 0:
                    aux_term = aux_loss if aux_loss.numel() == 1 else aux_loss.mean()
                    loss = loss + aux_term

                total_loss += loss.item()
                total_perplexity += torch.exp(loss).item()
                steps += 1

                progress_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})

                if max_steps is not None and i >= max_steps:
                    break

        avg_loss = total_loss / steps
        avg_perplexity = total_perplexity / steps

        return avg_loss, avg_perplexity

    def _save_checkpoint(self, epoch: int, step: int, global_step: int, **kwargs):
        if not self._ckpt_manager:
            return

        tag = f"epoch_{epoch}_step_{step}"
        self._ckpt_manager.save_checkpoint(
            self._engine,
            step=global_step,
            tag=tag,
            client_state={
                "epoch": epoch,
                "step": step,
                "global_step": global_step,
            }
            | kwargs,
        )

    def _generate(self):
        pass

    def _validate_kernel_policy(self, require_fused_kernels: bool):
        if require_fused_kernels and not HAS_TRITON:
            raise RuntimeError(
                "require_fused_kernels=true but required Triton kernels are unavailable "
                f"(HAS_TRITON={HAS_TRITON})."
            )

    def _cleanup(self):
        self._step_profiler.deactivate()
