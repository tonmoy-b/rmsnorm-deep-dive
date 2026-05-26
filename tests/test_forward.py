import pytest
import torch
import rmsnorm


def reference_rmsnorm(x, weight, eps):
    # Match the math of the kernel exactly: fp32 accumulation.
    orig_dtype = x.dtype
    x_f = x.float()
    var = x_f.pow(2).mean(-1, keepdim=True)
    y = x_f * torch.rsqrt(var + eps)
    y = y * weight.float()
    return y.to(orig_dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(4, 4096), (2, 8, 4096), (1, 8192)])
def test_forward_matches_reference(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    weight = torch.randn(shape[-1], device="cuda", dtype=dtype)
    eps = 1e-6

    y, rrms = rmsnorm.forward(x, weight, eps)
    y_ref = reference_rmsnorm(x, weight, eps)

    # Tolerances calibrated per dtype.
    if dtype == torch.float32:
        atol, rtol = 1e-5, 1e-5
    elif dtype == torch.float16:
        atol, rtol = 1e-2, 1e-2
    else:  # bfloat16
        atol, rtol = 3e-2, 3e-2

    max_err = (y - y_ref).abs().max().item()
    print(f"\n  shape={shape} dtype={dtype} max_abs_err={max_err:.4e}")
    torch.testing.assert_close(y, y_ref, atol=atol, rtol=rtol)


if __name__ == "__main__":
    # Allow `python tests/test_forward.py` for a quick smoke run.
    test_forward_matches_reference(torch.float16, (4, 4096))
    print("OK")