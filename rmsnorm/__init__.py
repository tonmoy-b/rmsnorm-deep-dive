import torch
from . import _C


class _RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        x = x.contiguous()
        weight = weight.contiguous()
        y, rrms = _C.forward(x, weight, eps)
        ctx.save_for_backward(x, weight, rrms)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, weight, rrms = ctx.saved_tensors
        grad_y = grad_y.contiguous()
        dx, dg = _C.backward(grad_y, x, weight, rrms)
        # return one grad per forward input so -> (x, weight, eps).
        # eps is not differentiable so -> None.
        return dx, dg, None


def rmsnorm(x, weight, eps=1e-6):
    """RMSNorm with custom CUDA forward/backward. Differentiable."""
    return _RMSNormFunction.apply(x, weight, eps)


# basic access to the kernels for testing/benchmarking.
forward = _C.forward
backward = _C.backward