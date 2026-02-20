# Test Configurations Summary

**Test 2: 20-step Initialization Setup**
This is a short 20-step training run designed strictly to establish a deterministic checkpoint and save an initial model state (`model_init.pt`). This checkpoint serves as the starting weight reference for all subsequent experiments.

**Test 3: Standard Embedding Baseline**
This is a comprehensive 1000-step baseline training test that runs the standard, non-reversible model architecture with conventional token embeddings. It explicitly loads the initialized weights from Test 2 to ensure a fair comparison.

**Test 4: Kronecker Embeddings**
This test modifies the model from Test 3 by replacing the standard embeddings with Kronecker embeddings and runs for 1000 steps. It retains the standard, non-reversible model backbone.

**Test 5: Reversible Architecture Activation**
This test preserves the Kronecker embeddings from Test 4 but swaps the core backbone to use the memory-saving Reversible Activation (`ReversibleMidpointStack`). It runs for 1000 steps to baseline the reversible model performance.

**Test 6: Fused Triton GSA Kernel**
This test takes the Reversible+Kronecker setup from Test 5 and introduces a highly optimized Triton kernel for the Gated Sparse Attention (GSA) layer. Since this represents a significant architectural change, it runs for a shorter 500-step validation.

**Test 7: Fused Triton GSA + DeltaNet (FLA)**
Building on Test 6, this test adds a second fused optimization path for the DeltaNet layers using the `fla_gated_delta_rule` from the Fast Linear Attention (FLA) package. The duration is increased back to 1000 steps to baseline total combined throughput. 

**Test 8: Additional Liger Kernels (MLP & CE)**
This test layers further optimizations onto Test 7 by replacing the standard PyTorch MLPs and language modeling loss with highly optimized `LigerSwiGLUMLP` and `LigerFusedLinearCrossEntropyLoss` implementations for 1000 steps.

**Test 9: 3000-Step DeepSpeed Checkpoint Resume**
This test does not introduce new optimizations, but instead acts as a validation that DeepSpeed can successfully resume the heavily-optimized Test 8 state from an epoch checkpoint and stably train it up to 3000 steps.

**Test 10: 3B-Class MoE Profile Conversion**
This test transforms the previous 1B dense Reversible+Kronecker backbone into a 3B-class Mixture of Experts (MoE) configuration with a `DDDGDDDG` mixed-attention sequence. It routes 40 Expert slots (20 Real, 20 Null) and runs as a short 100-step architectural check.

**Test 11: Fused MoE Backend (Grouped GEMM)**
This test takes the established 3B-class MoE from Test 10 and further pushes throughput by replacing standard PyTorch MoE operations with an optimized Fused MoE kernel path (`grouped_gemm`). It extends back up to a full 1000-step scaling baseline.
