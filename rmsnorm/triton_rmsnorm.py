import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fwd_kernel(
    x_ptr,        # *T, input  (N, H)
    g_ptr,        # *T, weight (H,)
    y_ptr,        # *T, output (N, H)
    rrms_ptr,     # *fp32, cached 1/rms (N,)
    stride_row,   # row stride of x/y (= H for contiguous)
    H,            # int, hidden dim
    eps,          # fp32
    BLOCK_H: tl.constexpr,   # power-of-two >= H
):
    # One program instance / block per row.
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_row
    y_row = y_ptr + row * stride_row

    # Column offsets covered, masked to H.
    cols = tl.arange(0, BLOCK_H)
    mask = cols < H

    # Load row, upcast to fp32 for reduction.
    x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)

    # Sum of squares --> mean --> reciprocal rms.
    sum_sq = tl.sum(x * x, axis=0)
    mean_sq = sum_sq / H
    rrms = 1.0 / tl.sqrt(mean_sq + eps)

    tl.store(rrms_ptr + row, rrms)

    # normalize then scale.
    g = tl.load(g_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rrms * g

    tl.store(y_row + cols, y.to(y_row.dtype.element_ty), mask=mask)


def rmsnorm_forward_triton(x, weight, eps=1e-6):
    """Triton RMSNorm forward. Returns (y, rrms). Mirrors CUDA op's contract."""
    x = x.contiguous()
    weight = weight.contiguous()

    H = x.shape[-1]
    x_2d = x.view(-1, H)
    N = x_2d.shape[0]

    y = torch.empty_like(x_2d)
    rrms = torch.empty((N,), device=x.device, dtype=torch.float32)

    # BLOCK_H must be a power of two >= H, so that the entire row fits in one program.
    BLOCK_H = triton.next_power_of_2(H)

    # heuristic -> more warps for wider rows (Note: Triton should be mapping this to threads).
    num_warps = 4
    if BLOCK_H >= 2048:
        num_warps = 8
    if BLOCK_H >= 4096:
        num_warps = 16

    grid = (N,)
    _rmsnorm_fwd_kernel[grid](
        x_2d, weight, y, rrms,
        x_2d.stride(0),
        H, eps,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
    )

    return y.view_as(x), rrms