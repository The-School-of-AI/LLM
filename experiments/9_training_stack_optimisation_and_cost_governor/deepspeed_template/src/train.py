"""
Training utilities for DeepSpeed.

This module contains training, evaluation, and inference functions
for training language models with DeepSpeed optimization.
"""

import torch
from tqdm import tqdm


def train_epoch(model_engine, train_loader, epoch, max_steps=None, log_interval=10):
    """
    Train the model for one epoch.

    Args:
        model_engine: DeepSpeed model engine
        train_loader: DataLoader for training data
        epoch: Current epoch number
        max_steps: Maximum number of steps per epoch (None for full epoch)
        log_interval: Log every N steps

    Returns:
        Average training loss for the epoch
    """
    model_engine.train()
    total_loss = 0
    steps = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for i, batch in enumerate(progress_bar):
        # Move batch to device
        input_ids = batch["input_ids"].to(model_engine.device)
        attention_mask = batch["attention_mask"].to(model_engine.device)
        labels = batch["labels"].to(model_engine.device)

        # Forward pass
        outputs = model_engine(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass
        model_engine.backward(loss)

        # Update weights
        model_engine.step()

        # Track metrics
        total_loss += loss.item()
        steps += 1

        # Update progress bar
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Log periodically
        if i % log_interval == 0:
            print(f"Epoch {epoch}, Step {i}, Loss: {loss.item():.4f}")

        # Early stopping for demo/debugging
        if max_steps is not None and i >= max_steps:
            break

    avg_loss = total_loss / steps
    print(f"Epoch {epoch} - Training Average Loss: {avg_loss:.4f}")

    return avg_loss


def evaluate(model_engine, data_loader, phase="Evaluation", max_steps=None):
    """
    Evaluate the model on a dataset.

    Args:
        model_engine: DeepSpeed model engine
        data_loader: DataLoader for evaluation data
        phase: Name of the evaluation phase (for logging)
        max_steps: Maximum number of steps (None for full evaluation)

    Returns:
        Tuple of (average_loss, average_perplexity)
    """
    model_engine.eval()
    total_loss = 0
    total_perplexity = 0
    steps = 0

    progress_bar = tqdm(data_loader, desc=phase)

    with torch.no_grad():
        for i, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(model_engine.device)
            attention_mask = batch["attention_mask"].to(model_engine.device)
            labels = batch["labels"].to(model_engine.device)

            # Forward pass
            outputs = model_engine(
                input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss

            # Track metrics
            total_loss += loss.item()
            total_perplexity += torch.exp(loss).item()
            steps += 1

            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Early stopping for demo/debugging
            if max_steps is not None and i >= max_steps:
                break

    avg_loss = total_loss / steps
    avg_perplexity = total_perplexity / steps

    print(f"{phase} - Avg Loss: {avg_loss:.4f}, Avg Perplexity: {avg_perplexity:.4f}")

    return avg_loss, avg_perplexity


def generate_text(
    model_engine,
    tokenizer,
    prompt="The history of artificial intelligence begins with",
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.92,
):
    """
    Generate text using the trained model.

    Args:
        model_engine: DeepSpeed model engine
        tokenizer: Tokenizer for encoding/decoding
        prompt: Input prompt for generation
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (lower = more conservative)
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter

    Returns:
        Dictionary with 'prompt', 'full_text', and 'generated_text'
    """
    model_engine.eval()

    print(f'\nGenerating text from prompt: "{prompt}"')

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model_engine.device)
    attention_mask = inputs["attention_mask"].to(model_engine.device)

    print(f"Input tokens: {input_ids.shape[1]}")

    # Generate
    with torch.no_grad():
        output_ids = model_engine.module.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode
    input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract generated portion
    generated_text = (
        full_output[len(input_text) :].strip()
        if len(full_output) > len(input_text)
        else ""
    )

    print(f"\nGenerated {output_ids.shape[1] - input_ids.shape[1]} new tokens")
    print(f"\nFull Output:\n{full_output}")
    print(f"\nGenerated Continuation:\n{generated_text}")

    return {
        "prompt": input_text,
        "full_text": full_output,
        "generated_text": generated_text,
    }


def save_checkpoint(model_engine, output_dir, tag="final"):
    """
    Save model checkpoint.

    Args:
        model_engine: DeepSpeed model engine
        output_dir: Directory to save checkpoint
        tag: Tag for the checkpoint
    """
    print(f"Saving checkpoint to {output_dir} with tag '{tag}'")
    model_engine.save_checkpoint(output_dir, tag=tag)
    print("Checkpoint saved successfully")


def load_checkpoint(model_engine, checkpoint_dir, tag="final"):
    """
    Load model checkpoint.

    Args:
        model_engine: DeepSpeed model engine
        checkpoint_dir: Directory containing checkpoint
        tag: Tag of the checkpoint to load

    Returns:
        The loaded checkpoint metadata
    """
    print(f"Loading checkpoint from {checkpoint_dir} with tag '{tag}'")
    _, client_sd = model_engine.load_checkpoint(checkpoint_dir, tag=tag)
    print("Checkpoint loaded successfully")
    return client_sd
