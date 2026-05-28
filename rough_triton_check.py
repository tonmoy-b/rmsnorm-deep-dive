import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def main():
    print("triton version:", triton.__version__)
    a = torch.randn(1024, device="cuda")
    b = torch.randn(1024, device="cuda")
    c = torch.empty_like(a)
    add_kernel[(1,)](a, b, c, 1024, BLOCK=1024)
    torch.cuda.synchronize()
    err = (c - (a + b)).abs().max().item()
    print("triton kernel ran, max err:", err)


if __name__ == "__main__":
    main()