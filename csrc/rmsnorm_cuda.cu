#include "rmsnorm.h"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <type_traits>

// ---------------------------------------------------------------------------
// Scalar conversions: load in storage dtype, accumulate in fp32.
// ---------------------------------------------------------------------------
template <typename T> __device__ __forceinline__ float to_float(T x);
template <> __device__ __forceinline__ float to_float<float>(float x) { return x; }
template <> __device__ __forceinline__ float to_float<__half>(__half x) { return __half2float(x); }
template <> __device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename T> __device__ __forceinline__ T from_float(float x);
template <> __device__ __forceinline__ float from_float<float>(float x) { return x; }
template <> __device__ __forceinline__ __half from_float<__half>(float x) { return __float2half(x); }
template <> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

// ---------------------------------------------------------------------------
// Warp / block reductions.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float v) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(0xffffffff, v, offset);
  }
  return v;
}

template <int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float v, float* smem) {
  constexpr int WARPS = BLOCK_SIZE / 32;
  const int lane = threadIdx.x & 31;
  const int wid  = threadIdx.x >> 5;

  v = warp_reduce_sum(v);
  if (lane == 0) smem[wid] = v;
  __syncthreads();

  v = (threadIdx.x < WARPS) ? smem[lane] : 0.0f;
  if (wid == 0) v = warp_reduce_sum(v);

  if (threadIdx.x == 0) smem[0] = v;
  __syncthreads();
  return smem[0];
}

// ---------------------------------------------------------------------------
// 16-byte vector type per scalar.
// ---------------------------------------------------------------------------
template <typename T, int VEC> struct VecTypeHelper;
template <> struct VecTypeHelper<float,         4> { using type = float4; };
template <> struct VecTypeHelper<__half,        8> { using type = uint4;  };
template <> struct VecTypeHelper<__nv_bfloat16, 8> { using type = uint4;  };

// ---------------------------------------------------------------------------
// Forward kernel: one block per row.
// ---------------------------------------------------------------------------
template <typename T, int BLOCK_SIZE, int VEC>
__global__ void rmsnorm_forward_kernel(
    const T* __restrict__ x,
    const T* __restrict__ g,
    T* __restrict__ y,
    float* __restrict__ rrms,
    int H,
    float eps) {

  using VecT = typename VecTypeHelper<T, VEC>::type;

  const int row = blockIdx.x;
  const T* x_row = x + row * H;
  T*       y_row = y + row * H;

  __shared__ float smem[BLOCK_SIZE / 32];

  const VecT* x_vec = reinterpret_cast<const VecT*>(x_row);
  const int H_vec = H / VEC;

  // Pass 1: sum of squares (fp32 accumulation)
  float sum_sq = 0.0f;
  for (int i = threadIdx.x; i < H_vec; i += BLOCK_SIZE) {
    VecT v = x_vec[i];
    const T* v_as_T = reinterpret_cast<const T*>(&v);
    #pragma unroll
    for (int k = 0; k < VEC; ++k) {
      float f = to_float<T>(v_as_T[k]);
      sum_sq += f * f;
    }
  }

  sum_sq = block_reduce_sum<BLOCK_SIZE>(sum_sq, smem);
  const float mean_sq = sum_sq / static_cast<float>(H);
  const float r = rsqrtf(mean_sq + eps);

  if (threadIdx.x == 0) rrms[row] = r;

  // Pass 2: normalize and scale
  VecT* y_vec = reinterpret_cast<VecT*>(y_row);
  const VecT* g_vec = reinterpret_cast<const VecT*>(g);

  for (int i = threadIdx.x; i < H_vec; i += BLOCK_SIZE) {
    VecT vx = x_vec[i];
    VecT vg = g_vec[i];
    VecT vy;

    const T* vx_T = reinterpret_cast<const T*>(&vx);
    const T* vg_T = reinterpret_cast<const T*>(&vg);
    T*       vy_T = reinterpret_cast<T*>(&vy);

    #pragma unroll
    for (int k = 0; k < VEC; ++k) {
      float fx = to_float<T>(vx_T[k]);
      float fg = to_float<T>(vg_T[k]);
      vy_T[k]  = from_float<T>(fx * r * fg);
    }
    y_vec[i] = vy;
  }
}

// ---------------------------------------------------------------------------
// Launch helpers (kept out of the dispatch lambda so we don't put #define
// inside a macro argument list -- forbidden by conforming preprocessor).
// ---------------------------------------------------------------------------
template <typename T, int BLOCK_SIZE, int VEC>
static inline void launch_rmsnorm_forward(
    dim3 grid, dim3 block, cudaStream_t stream,
    const T* x_ptr, const T* g_ptr, T* y_ptr, float* rrms_ptr,
    int H, float eps) {
  rmsnorm_forward_kernel<T, BLOCK_SIZE, VEC><<<grid, block, 0, stream>>>(
      x_ptr, g_ptr, y_ptr, rrms_ptr, H, eps);
}

template <typename T>
static void dispatch_rmsnorm_forward(
    dim3 grid, dim3 block, cudaStream_t stream,
    const T* x_ptr, const T* g_ptr, T* y_ptr, float* rrms_ptr,
    int H, float eps, int block_size, int vec) {

  if (vec == 4) {
    if      (block_size == 1024) launch_rmsnorm_forward<T, 1024, 4>(grid, block, stream, x_ptr, g_ptr, y_ptr, rrms_ptr, H, eps);
    else if (block_size == 512)  launch_rmsnorm_forward<T, 512,  4>(grid, block, stream, x_ptr, g_ptr, y_ptr, rrms_ptr, H, eps);
    else if (block_size == 256)  launch_rmsnorm_forward<T, 256,  4>(grid, block, stream, x_ptr, g_ptr, y_ptr, rrms_ptr, H, eps);
    else                          launch_rmsnorm_forward<T, 128,  4>(grid, block, stream, x_ptr, g_ptr, y_ptr, rrms_ptr, H, eps);
  } else {
    if      (block_size == 1024) launch_rmsnorm_forward<T, 1024, 8>(grid, block, stream, x_ptr, g_ptr, y_ptr, rrms_ptr, H, eps);
    else if (block_size == 512)  launch_rmsnorm_forward<T, 512,  8>(grid, block, stream, x_ptr, g_ptr, y_ptr, rrms_ptr, H, eps);
    else if (block_size == 256)  launch_rmsnorm_forward<T, 256,  8>(grid, block, stream, x_ptr, g_ptr, y_ptr, rrms_ptr, H, eps);
    else                          launch_rmsnorm_forward<T, 128,  8>(grid, block, stream, x_ptr, g_ptr, y_ptr, rrms_ptr, H, eps);
  }
}

// ---------------------------------------------------------------------------
// Host entry point.
// ---------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> rmsnorm_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    double eps) {

  const int H = x.size(-1);
  const int N = x.numel() / H;

  auto y    = torch::empty_like(x);
  auto rrms = torch::empty({N}, x.options().dtype(torch::kFloat32));

  const int vec = (x.element_size() == 4) ? 4 : 8;
  TORCH_CHECK(H % vec == 0,
              "H must be divisible by ", vec, " (got ", H, ")");

  int block_size = 1024;
  while (block_size > 32 && block_size > (H / vec)) block_size >>= 1;

  const dim3 grid(N);
  const dim3 block(block_size);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_SWITCH(x.scalar_type(), "rmsnorm_forward_cuda",
    AT_DISPATCH_CASE(at::ScalarType::Float, [&] {
      using T = float;
      dispatch_rmsnorm_forward<T>(
          grid, block, stream,
          reinterpret_cast<const T*>(x.data_ptr<float>()),
          reinterpret_cast<const T*>(weight.data_ptr<float>()),
          reinterpret_cast<T*>(y.data_ptr<float>()),
          rrms.data_ptr<float>(),
          H, static_cast<float>(eps), block_size, vec);
    })
    AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
      using T = __half;
      dispatch_rmsnorm_forward<T>(
          grid, block, stream,
          reinterpret_cast<const T*>(x.data_ptr<at::Half>()),
          reinterpret_cast<const T*>(weight.data_ptr<at::Half>()),
          reinterpret_cast<T*>(y.data_ptr<at::Half>()),
          rrms.data_ptr<float>(),
          H, static_cast<float>(eps), block_size, vec);
    })
    AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
      using T = __nv_bfloat16;
      dispatch_rmsnorm_forward<T>(
          grid, block, stream,
          reinterpret_cast<const T*>(x.data_ptr<at::BFloat16>()),
          reinterpret_cast<const T*>(weight.data_ptr<at::BFloat16>()),
          reinterpret_cast<T*>(y.data_ptr<at::BFloat16>()),
          rrms.data_ptr<float>(),
          H, static_cast<float>(eps), block_size, vec);
    })
  );

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, rrms};
}