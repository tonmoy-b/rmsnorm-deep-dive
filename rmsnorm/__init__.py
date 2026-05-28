import torch
from torch import Tensor
from . import _C


# define op and bind CUDA implementation.
@torch.library.custom_op("rmsnorm::forward", mutates_args=())
def _rmsnorm_fwd(x: Tensor, weight: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    y, rrms = _C.forward(x.contiguous(), weight.contiguous(), eps)
    return y, rrms


@_rmsnorm_fwd.register_fake
def _rmsnorm_fwd_fake(x, weight, eps):
    y = torch.empty_like(x)
    N = x.numel() // x.shape[-1]
    rrms = x.new_empty((N,), dtype=torch.float32)
    return y, rrms


# backward op
@torch.library.custom_op("rmsnorm::backward", mutates_args=())
def _rmsnorm_bwd(dy: Tensor, x: Tensor, weight: Tensor, rrms: Tensor) -> tuple[Tensor, Tensor]:
    dx, dg = _C.backward(dy.contiguous(), x.contiguous(),
                         weight.contiguous(), rrms.contiguous())
    return dx, dg


@_rmsnorm_bwd.register_fake
def _rmsnorm_bwd_fake(dy, x, weight, rrms):
    dx = torch.empty_like(x)
    dg = torch.empty_like(weight)
    return dx, dg


# put the forward op into autograd with setup_context / backward.
def _backward(ctx, grad_y, grad_rrms):
    x, weight, rrms = ctx.saved_tensors
    dx, dg = torch.ops.rmsnorm.backward(grad_y, x, weight, rrms)
    return dx, dg, None  # (x, weight, eps) -> eps has no grad


def _setup_context(ctx, inputs, output):
    x, weight, eps = inputs
    y, rrms = output
    ctx.save_for_backward(x, weight, rrms)


_rmsnorm_fwd.register_autograd(_backward, setup_context=_setup_context)


# Public API for access
def rmsnorm(x, weight, eps=1e-6):
    """RMSNorm via the registered custom op. Differentiable AND compile-friendly."""
    y, _ = torch.ops.rmsnorm.forward(x, weight, eps)
    return y


# kernel access for testing/benchmarking.
forward = _C.forward
backward = _C.backward