"""
Quick Test for Memory Profiler
===============================

Simple test to verify the profiler module works correctly.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from memory_profiler import MemoryProfiler, ProfilerConfig


def test_profiler():
    """Test basic profiler functionality."""

    print("=" * 80)
    print("MEMORY PROFILER TEST")
    print("=" * 80)

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA Available: {cuda_available}")
    print(f"PyTorch Version: {torch.__version__}")

    # Create profiler config
    config = ProfilerConfig(
        output_dir="./test_profiler_logs",
        profile_memory=True,
        profile_cpu=True,
        profile_cuda=cuda_available,
        wait_steps=2,
        warmup_steps=2,
        active_steps=5,
        repeat=1,
    )

    print("\nProfiler Configuration:")
    print(f"  Output: {config.output_dir}")
    print(f"  Wait: {config.wait_steps} steps")
    print(f"  Warmup: {config.warmup_steps} steps")
    print(f"  Active: {config.active_steps} steps")
    print(
        f"  Total: {config.wait_steps + config.warmup_steps + config.active_steps} steps"
    )

    # Create profiler
    profiler = MemoryProfiler(config)

    # Create simple model
    device = torch.device("cuda" if cuda_available else "cpu")
    model = torch.nn.Sequential(
        torch.nn.Linear(512, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, 512)
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print(f"\nModel Device: {device}")
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Start profiling
    profiler.start()

    print("\nRunning training steps...")
    total_steps = profiler.get_total_steps()

    for step in range(total_steps):
        # Forward pass
        with profiler.profile_section("forward"):
            x = torch.randn(8, 512).to(device)
            output = model(x)
            loss = output.mean()

        # Backward pass
        with profiler.profile_section("backward"):
            loss.backward()

        # Optimizer step
        with profiler.profile_section("optimizer"):
            optimizer.step()
            optimizer.zero_grad()

        # Step profiler
        profiler.step()

        if step % 2 == 0:
            print(f"  Step {step}/{total_steps} - Loss: {loss.item():.4f}")

        # Check if should stop
        if profiler.should_stop(step):
            break

    # Stop profiling
    profiler.stop()

    # Print summary
    print("\n" + "=" * 80)
    profiler.print_summary()

    # Export results
    profiler.export_chrome_trace()
    profiler.export_stacks()

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {profiler.output_dir}")
    print("\nTo view results:")
    print(f"  1. TensorBoard: tensorboard --logdir={profiler.tensorboard_dir}")
    print(
        f"  2. Chrome trace: chrome://tracing → Load {profiler.output_dir}/memory_profile.json"
    )
    print(f"  3. Stack traces: cat {profiler.output_dir}/stack_trace.txt")
    print("=" * 80)


if __name__ == "__main__":
    test_profiler()
