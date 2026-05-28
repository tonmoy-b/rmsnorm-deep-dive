#include "rmsnorm.h"
#include <torch/extension.h>

#define CHECK_CUDA(x)        TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)       CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::tuple<torch::Tensor, torch::Tensor> rmsnorm_forward(
    torch::Tensor x, torch::Tensor weight, double eps) {
  CHECK_INPUT(x);
  CHECK_INPUT(weight);
  TORCH_CHECK(x.size(-1) == weight.size(0), "weight size must match last dim of x");
  TORCH_CHECK(x.scalar_type() == weight.scalar_type(), "x and weight must have same dtype");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32 ||
              x.scalar_type() == torch::kFloat16 ||
              x.scalar_type() == torch::kBFloat16, "only fp32/fp16/bf16 supported");
  return rmsnorm_forward_cuda(x, weight, eps);
}

std::tuple<torch::Tensor, torch::Tensor> rmsnorm_backward(
    torch::Tensor dy, torch::Tensor x, torch::Tensor weight, torch::Tensor rrms) {
  CHECK_INPUT(dy);
  CHECK_INPUT(x);
  CHECK_INPUT(weight);
  CHECK_INPUT(rrms);
  return rmsnorm_backward_cuda(dy, x, weight, rrms);
}

PYBIND11_MODULE(_C, m) {
  m.def("forward",  &rmsnorm_forward,  "RMSNorm forward (CUDA)");
  m.def("backward", &rmsnorm_backward, "RMSNorm backward (CUDA)");
}