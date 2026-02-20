import torch
from triton_sparse_attn import triton_sparse_attention, pytorch_sparse_attention
from triton_sparse_attn_v2 import triton_sparse_attention_v2

def test_correctness():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running correctness test on {device}...")
    
    B, H, T_q, T_kv, D, k_sel = 2, 4, 128, 128, 64, 32
    
    torch.manual_seed(42)
    q = torch.randn(B, T_q, H, D, device=device, dtype=torch.float32, requires_grad=True)
    k = torch.randn(B, T_kv, H, D, device=device, dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, T_kv, H, D, device=device, dtype=torch.float32, requires_grad=True)
    
    # 1. Normal Test
    indices = torch.randint(0, T_kv, (B, H, T_q, k_sel), device=device, dtype=torch.int64)
    mask = (torch.rand(B, H, T_q, k_sel, device=device) < 0.9).float()
    scale = 1.0 / (D ** 0.5)
    
    # Run PyTorch Fallback
    out_pt = pytorch_sparse_attention(q, k, v, indices, mask, scale)
    grad_out = torch.randn_like(out_pt)
    out_pt.backward(grad_out, retain_graph=True)
    
    dq_pt, q.grad = q.grad.clone(), None
    dk_pt, k.grad = k.grad.clone(), None
    dv_pt, v.grad = v.grad.clone(), None
    
    # Run Triton V1
    out_tr = triton_sparse_attention(q, k, v, indices, mask, scale, use_triton_backward=True)
    out_tr.backward(grad_out)
    
    dq_tr = q.grad.clone()
    dk_tr = k.grad.clone()
    dv_tr = v.grad.clone()
    
    def check_max_diff(name, t1, t2):
        diff = (t1 - t2).abs().max().item()
        print(f"Max diff {name}: {diff:.6f}")
        return diff
    
    print("\n--- V1: Normal Case ---")
    check_max_diff("Out", out_pt, out_tr)
    check_max_diff("dQ", dq_pt, dq_tr)
    check_max_diff("dK", dk_pt, dk_tr)
    check_max_diff("dV", dv_pt, dv_tr)

    # ── V2: Normal Test ──
    q.grad, k.grad, v.grad = None, None, None
    out_v2 = triton_sparse_attention_v2(q, k, v, indices, mask, scale, use_triton_backward=True)
    out_v2.backward(grad_out)

    dq_v2 = q.grad.clone()
    dk_v2 = k.grad.clone()
    dv_v2 = v.grad.clone()

    print("\n--- V2 (tl.dot): Normal Case ---")
    check_max_diff("Out", out_pt, out_v2)
    check_max_diff("dQ", dq_pt, dq_v2)
    check_max_diff("dK", dk_pt, dk_v2)
    check_max_diff("dV", dv_pt, dv_v2)

    # 2. Edge Case: Fully masked row & out-of-bounds indices
    print("\n--- V1: Edge Case ---")
    q.grad, k.grad, v.grad = None, None, None
    mask[:, :, 0, :] = 0.0 # Row 0 is fully masked!
    indices[:, :, 1, 0] = T_kv + 10 # Out of bounds index
    
    # We must clear gradients and re-run PyTorch with new mask/indices
    out_pt = pytorch_sparse_attention(q, k, v, indices, mask, scale)
    out_pt.backward(grad_out, retain_graph=True)
    
    dq_pt, q.grad = q.grad.clone(), None
    dk_pt, k.grad = k.grad.clone(), None
    dv_pt, v.grad = v.grad.clone(), None
    
    try:
        out_tr = triton_sparse_attention(q, k, v, indices, mask, scale, use_triton_backward=True)
        out_tr.backward(grad_out)
        
        dq_tr = q.grad.clone()
        dk_tr = k.grad.clone()
        dv_tr = v.grad.clone()
        
        check_max_diff("Out", out_pt, out_tr)
        check_max_diff("dQ", dq_pt, dq_tr)
        check_max_diff("dK", dk_pt, dk_tr)
        check_max_diff("dV", dv_pt, dv_tr)
        print("V1 Edge case survived without crashing!")
    except Exception as e:
        print(f"V1 Edge case failed: {e}")

    # ── V2: Edge Case ──
    print("\n--- V2 (tl.dot): Edge Case ---")
    q.grad, k.grad, v.grad = None, None, None
    try:
        out_v2 = triton_sparse_attention_v2(q, k, v, indices, mask, scale, use_triton_backward=True)
        out_v2.backward(grad_out)

        dq_v2 = q.grad.clone()
        dk_v2 = k.grad.clone()
        dv_v2 = v.grad.clone()

        check_max_diff("Out", out_pt, out_v2)
        check_max_diff("dQ", dq_pt, dq_v2)
        check_max_diff("dK", dk_pt, dk_v2)
        check_max_diff("dV", dv_pt, dv_v2)
        print("V2 Edge case survived without crashing!")
    except Exception as e:
        print(f"V2 Edge case failed: {e}")

if __name__ == '__main__':
    test_correctness()
