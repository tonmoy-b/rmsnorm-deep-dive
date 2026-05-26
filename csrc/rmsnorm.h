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