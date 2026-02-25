"""
triton_cross_entropy.py — Optimized Fused Linear + CE Kernel
"""
import torch
import triton
import triton.language as tl

_MAX_FUSED_SIZE = 32768

@triton.jit
def _liger_cross_entropy_kernel(
    X_ptr, X_stride, Y_ptr, Y_stride, loss_ptr, loss_stride,
    n_cols, n_non_ignore, ignore_index, reduction: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, HAS_GRADIENTS: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)
    if y == ignore_index:
        if HAS_GRADIENTS:
            X_ptr += program_id * X_stride
            for i in range(0, n_cols, BLOCK_SIZE):
                X_offsets = i + tl.arange(0, BLOCK_SIZE)
                tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        return
    X_ptr += program_id * X_stride
    loss_ptr += program_id * loss_stride
    m = float("-inf")
    d = 0.0
    ori_X_y = tl.load(X_ptr + y).to(tl.float32)
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols, other=float("-inf")).to(tl.float32)
        block_max = tl.max(X_block)
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new
    lse = m + tl.log(d)
    if HAS_GRADIENTS:
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols, other=float("-inf")).to(tl.float32)
            grad = tl.exp(X_block - m) / d
            grad = tl.where(X_offsets == y, grad - 1.0, grad)
            if reduction == "mean":
                grad = grad / n_non_ignore
            tl.store(X_ptr + X_offsets, grad, mask=X_offsets < n_cols)
    tl.debug_barrier()
    loss = lse - ori_X_y
    if reduction == "mean":
        loss = loss / n_non_ignore
    tl.store(loss_ptr, loss)

def _fused_linear_ce_forward(_input, weight, target, ignore_index, reduction, max_chunk_bytes):
    BT, H = _input.shape
    V = weight.shape[0]
    
    # Auto chunk size - aiming for 4GB chunks for max throughput
    elem_size = 4 
    max_elems = max_chunk_bytes // (V * elem_size)
    chunk_size = max(1, min(BT, int(max_elems)))
    
    # Align to power of 2 for GPU efficiency
    _cs = 1
    while _cs * 2 <= chunk_size: _cs *= 2
    chunk_size = min(_cs, BT)
    
    BLOCK_SIZE = min(_MAX_FUSED_SIZE, triton.next_power_of_2(V))
    n_non_ignore = max(int((target != ignore_index).sum().item()), 1)
    
    grad_input = torch.zeros_like(_input) if _input.requires_grad else None
    grad_weight = torch.zeros_like(weight) if weight.requires_grad else None
    loss_accum = torch.zeros(1, device=_input.device, dtype=torch.float32)

    # Pre-contiguous weight transpose for speed
    weight_T = weight.T.contiguous()

    for start in range(0, BT, chunk_size):
        end = min(start + chunk_size, BT)
        C = end - start
        
        h_chunk = _input[start:end]
        t_chunk = target[start:end]
        
        # BF16 Matmul (Speed!) -> Cast to FP32 (Accuracy for CE)
        logits_chunk = (h_chunk @ weight_T).float().contiguous()
        
        loss_1d = torch.zeros(C, dtype=torch.float32, device=_input.device)
        
        _liger_cross_entropy_kernel[(C,)](
            logits_chunk, logits_chunk.stride(-2), t_chunk, t_chunk.stride(-1),
            loss_1d, loss_1d.stride(-1), V, n_non_ignore, ignore_index, reduction,
            BLOCK_SIZE, (grad_input is not None or grad_weight is not None),
        )
        
        loss_accum += loss_1d.sum()
        
        if grad_input is not None:
            grad_input[start:end].add_((logits_chunk.to(_input.dtype) @ weight))
        if grad_weight is not None:
            grad_weight.add_((logits_chunk.T.to(weight.dtype) @ h_chunk))
            
    return loss_accum.squeeze(), grad_input, grad_weight

class _FusedLinearCEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, _input, weight, target, ignore_index, reduction, max_chunk_gb):
        max_chunk_bytes = int(max_chunk_gb * 1024 * 1024 * 1024)
        loss, grad_input, grad_weight = _fused_linear_ce_forward(_input, weight, target, ignore_index, reduction, max_chunk_bytes)
        ctx.save_for_backward(grad_input, grad_weight)
        return loss
    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_weight = ctx.saved_tensors
        if not torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
             if grad_input is not None: grad_input *= grad_output
             if grad_weight is not None: grad_weight *= grad_output
        return grad_input, grad_weight, None, None, None, None

class FusedLinearCrossEntropyLoss(torch.nn.Module):
    def __init__(self, ignore_index=-100, reduction="mean", max_chunk_gb=16.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.max_chunk_gb = max_chunk_gb

    def forward(self, hidden_states, weight, target):
        return _FusedLinearCEFunction.apply(
            hidden_states, weight, target, self.ignore_index, self.reduction, self.max_chunk_gb
        )
