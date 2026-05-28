#pragma once
#include <torch/extension.h>

// Forward: returns (y, rrms). rrms is cached for backward.
std::tuple<torch::Tensor, torch::Tensor> rmsnorm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    double eps);

// CUDA implementation (defined in rmsnorm_cuda.cu).
std::tuple<torch::Tensor, torch::Tensor> rmsnorm_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    double eps);

// Backward: given dy and the cached forward state, returns (dx, dg).
std::tuple<torch::Tensor, torch::Tensor> rmsnorm_backward(
    torch::Tensor dy, torch::Tensor x, torch::Tensor weight, torch::Tensor rrms);

std::tuple<torch::Tensor, torch::Tensor> rmsnorm_backward_cuda(
    torch::Tensor dy, torch::Tensor x, torch::Tensor weight, torch::Tensor rrms);