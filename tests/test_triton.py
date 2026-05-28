import pytest
import torch
from rmsnorm.triton_rmsnorm import rmsnorm_forward_triton


def reference_rmsnorm(x, weight, eps):
    x_f = x.float()
    var = x_f.pow(2).mean(-1, keepdim=True)
    y = x_f * torch.rsqrt(var + eps)
    return (y * weight.float()).to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(4, 4096), (2, 8, 4096), (1, 8192), (3, 4000)])
def test_triton_forward_matches_reference(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    w = torch.randn(shape[-1], device="cuda", dtype=dtype)
    eps = 1e-6

    y, rrms = rmsnorm_forward_triton(x, w, eps)
    y_ref = reference_rmsnorm(x, w, eps)

    if dtype == torch.float32:
        atol, rtol = 1e-5, 1e-5
    elif dtype == torch.float16:
        atol, rtol = 1e-2, 1e-2
    else:
        atol, rtol = 3e-2, 3e-2

    max_err = (y - y_ref).abs().max().item()
    print(f"\n  shape={shape} dtype={dtype} max_abs_err={max_err:.4e}")
    torch.testing.assert_close(y, y_ref, atol=atol, rtol=rtol)