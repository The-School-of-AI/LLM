import torch

from triton_sparse_attn import triton_sparse_attention, pytorch_sparse_attention


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; skipping test.")
        return

    device = torch.device("cuda")
    B, H, T_q, T_kv, D, k_sel = 2, 4, 128, 128, 64, 32
    scale = 1.0 / (D ** 0.5)

    torch.manual_seed(42)
    q = torch.randn(B, T_q, H, D, device=device, dtype=torch.float32, requires_grad=True)
    k = torch.randn(B, T_kv, H, D, device=device, dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, T_kv, H, D, device=device, dtype=torch.float32, requires_grad=True)

    indices = torch.randint(0, T_kv, (B, H, T_q, k_sel), device=device, dtype=torch.int64)
    mask = (torch.rand(B, H, T_q, k_sel, device=device) < 0.9).float()

    # Baseline compare (normal case)
    out_pt = pytorch_sparse_attention(q, k, v, indices, mask, scale)
    grad_out = torch.randn_like(out_pt)
    out_pt.backward(grad_out, retain_graph=True)
    q.grad, k.grad, v.grad = None, None, None

    out_tr = triton_sparse_attention(q, k, v, indices, mask, scale, use_triton_backward=True)
    out_tr.backward(grad_out, retain_graph=True)
    q.grad, k.grad, v.grad = None, None, None

    # Case 1: masked-in OOB index must raise (sanitization behavior)
    idx_bad = indices.clone()
    mask_bad = mask.clone()
    idx_bad[:, :, 1, 0] = T_kv + 10
    mask_bad[:, :, 1, 0] = 1.0
    try:
        _ = triton_sparse_attention(q, k, v, idx_bad, mask_bad, scale, use_triton_backward=True)
        raise AssertionError("Expected ValueError for masked-in OOB index, but call succeeded.")
    except ValueError:
        print("Case 1 OK: masked-in OOB raises ValueError.")

    # Case 2: masked-out OOB index should survive and stay finite (crash-proofing behavior)
    idx_safe = indices.clone()
    mask_safe = mask.clone()
    idx_safe[:, :, 1, 0] = T_kv + 10
    mask_safe[:, :, 1, 0] = 0.0

    out_safe = triton_sparse_attention(q, k, v, idx_safe, mask_safe, scale, use_triton_backward=True)
    if not torch.isfinite(out_safe).all():
        raise AssertionError("Masked-out OOB case produced non-finite output.")
    out_safe.backward(grad_out)

    print("Case 2 OK: masked-out OOB survived without crashing and backward succeeded.")


if __name__ == "__main__":
    main()
