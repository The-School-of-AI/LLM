import time
import torch
from tqdm import tqdm

from src.moe_utils import try_get_moe_expert_counts, expert_imbalance_ratio


def _safe_get_lr(model_engine):
    try:
        return model_engine.optimizer.param_groups[0]["lr"]
    except Exception:
        return None


def _safe_gpu_mem_mb():
    if not torch.cuda.is_available():
        return None
    try:
        return int(torch.cuda.memory_allocated() / (1024 * 1024))
    except Exception:
        return None


def train_epoch(model_engine, train_loader, epoch, writer=None, log_interval=10, global_step_start=0):
    model_engine.train()
    total_loss = 0.0
    steps = 0
    global_step = global_step_start

    progress = tqdm(
        train_loader,
        desc=f"train epoch {epoch}",
        disable=(getattr(model_engine, "global_rank", 0) != 0),
    )

    for step, batch in enumerate(progress):
        t0 = time.time()

        input_ids = batch["input_ids"].to(model_engine.device)
        attention_mask = batch["attention_mask"].to(model_engine.device)
        labels = batch["labels"].to(model_engine.device)

        # forward
        tf0 = time.time()
        outputs = model_engine(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        tf_ms = (time.time() - tf0) * 1000

        # backward
        tb0 = time.time()
        model_engine.backward(loss)
        tb_ms = (time.time() - tb0) * 1000

        # step
        ts0 = time.time()
        model_engine.step()
        ts_ms = (time.time() - ts0) * 1000

        total_loss += loss.item()
        steps += 1

        # tokens/s (rough)
        dt = max(time.time() - t0, 1e-6)
        tok_s = input_ids.numel() / dt

        lr = _safe_get_lr(model_engine)
        mem_mb = _safe_gpu_mem_mb()

        # expert load
        expert_counts = try_get_moe_expert_counts(model_engine.module)
        imbalance = expert_imbalance_ratio(expert_counts)

        # tqdm line
        postfix = {"loss": f"{loss.item():.4f}", "tok/s": f"{tok_s:.0f}"}
        if lr is not None:
            postfix["lr"] = f"{lr:.2e}"
        if mem_mb is not None:
            postfix["memMB"] = f"{mem_mb}"
        if imbalance is not None:
            postfix["imb"] = f"{imbalance:.2f}"

        progress.set_postfix(postfix)

        # terminal log
        if step % log_interval == 0 and getattr(model_engine, "global_rank", 0) == 0:
            print(
                f"[train] epoch={epoch} step={step} gstep={global_step} "
                f"loss={loss.item():.4f} "
                f"lr={lr if lr is not None else 'NA'} "
                f"tok/s={tok_s:.0f} "
                f"mem_alloc={mem_mb if mem_mb is not None else 'NA'}MB "
                f"time(ms): fwd={tf_ms:.1f} bwd={tb_ms:.1f} step={ts_ms:.1f}"
            )

        # tensorboard
        if writer is not None and getattr(model_engine, "global_rank", 0) == 0:
            writer.add_scalar("train/loss", loss.item(), global_step)
            if lr is not None:
                writer.add_scalar("train/lr", lr, global_step)
            writer.add_scalar("train/tok_per_s", tok_s, global_step)
            if mem_mb is not None:
                writer.add_scalar("train/gpu_mem_mb", mem_mb, global_step)

            if expert_counts is not None:
                # histogram + imbalance
                writer.add_histogram("moe/expert_counts", expert_counts, global_step)
                if imbalance is not None:
                    writer.add_scalar("moe/imbalance_ratio", imbalance, global_step)

        global_step += 1

    avg_loss = total_loss / max(steps, 1)
    if getattr(model_engine, "global_rank", 0) == 0:
        print(f"[train] epoch={epoch} avg_loss={avg_loss:.4f}")

    return avg_loss, global_step


def evaluate(model_engine, data_loader, phase="eval", writer=None, global_step=None):
    model_engine.eval()
    total_loss = 0.0
    steps = 0

    progress = tqdm(
        data_loader,
        desc=phase,
        disable=(getattr(model_engine, "global_rank", 0) != 0),
    )

    with torch.no_grad():
        for batch in progress:
            input_ids = batch["input_ids"].to(model_engine.device)
            attention_mask = batch["attention_mask"].to(model_engine.device)
            labels = batch["labels"].to(model_engine.device)

            outputs = model_engine(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            total_loss += loss.item()
            steps += 1
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / max(steps, 1)
    ppl = float(torch.exp(torch.tensor(avg_loss)))

    if getattr(model_engine, "global_rank", 0) == 0:
        print(f"[{phase}] avg_loss={avg_loss:.4f} ppl={ppl:.2f}")
        if writer is not None and global_step is not None:
            writer.add_scalar(f"{phase}/loss", avg_loss, global_step)
            writer.add_scalar(f"{phase}/ppl", ppl, global_step)

    return avg_loss, ppl


def generate_text(model_engine, tokenizer, prompt, max_new_tokens=80):
    """
    Safe generation for MoE: min_capacity already set to 0 in the MoE layer,
    so small prompts won't crash.
    """
    model_engine.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model_engine.device)
    attention_mask = inputs["attention_mask"].to(model_engine.device)

    with torch.no_grad():
        out = model_engine.module.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if getattr(model_engine, "global_rank", 0) == 0:
        print("\n=== Generation ===")
        print(text)
    return text


def save_checkpoint(model_engine, output_dir, tag="final"):
    if getattr(model_engine, "global_rank", 0) == 0:
        print(f"[ckpt] saving -> {output_dir} tag={tag}")
    model_engine.save_checkpoint(output_dir, tag=tag)


def load_checkpoint(model_engine, checkpoint_dir, tag="final"):
    if getattr(model_engine, "global_rank", 0) == 0:
        print(f"[ckpt] loading <- {checkpoint_dir} tag={tag}")
    _, client_sd = model_engine.load_checkpoint(checkpoint_dir, tag=tag)
    return client_sd