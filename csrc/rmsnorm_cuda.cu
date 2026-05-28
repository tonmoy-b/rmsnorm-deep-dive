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
template <typename T>
__device__ __forceinline__ float to_float(T x);
template <>
__device__ __forceinline__ float to_float<float>(float x) { return x; }
template <>
__device__ __forceinline__ float to_float<__half>(__half x) { return __half2float(x); }
template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename T>
__device__ __forceinline__ T from_float(float x);
template <>
__device__ __forceinline__ float from_float<float>(float x) { return x; }
template <>
__device__ __forceinline__ __half from_float<__half>(float x) { return __float2half(x); }
template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

// ---------------------------------------------------------------------------
// Warp / block reductions.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float v)
{
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
  {
    v += __shfl_xor_sync(0xffffffff, v, offset);
  }
  return v;
}

template <int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float v, float *smem)
{
  constexpr int WARPS = BLOCK_SIZE / 32;
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;

  v = warp_reduce_sum(v);
  if (lane == 0)
    smem[wid] = v;
  __syncthreads();

  v = (threadIdx.x < WARPS) ? smem[lane] : 0.0f;
  if (wid == 0)
    v = warp_reduce_sum(v);

  if (threadIdx.x == 0)
    smem[0] = v;
  __syncthreads();
  return smem[0];
}

// ---------------------------------------------------------------------------
// 16-byte vector type per scalar.
// ---------------------------------------------------------------------------
template <typename T, int VEC>
struct VecTypeHelper;
template <>
struct VecTypeHelper<float, 4>
{
  using type = float4;
};
template <>
struct VecTypeHelper<__half, 8>
{
  using type = uint4;
};
template <>
struct VecTypeHelper<__nv_bfloat16, 8>
{
  using type = uint4;
};

// ---------------------------------------------------------------------------
// Forward kernel: one block per row.
// ---------------------------------------------------------------------------
template <typename T, int BLOCK_SIZE, int VEC>
__global__ void rmsnorm_forward_kernel(
    const T *__restrict__ x,
    const T *__restrict__ g,
    T *__restrict__ y,
    float *__restrict__ rrms,
    int H,
    float eps)
{

  using VecT = typename VecTypeHelper<T, VEC>::type;

  const int row = blockIdx.x;
  const T *x_row = x + row * H;
  T *y_row = y + row * H;

  __shared__ float smem[BLOCK_SIZE / 32];

  const VecT *x_vec = reinterpret_cast<const VecT *>(x_row);
  const int H_vec = H / VEC;

  // Pass 1: sum of squares (fp32 accumulation)
  float sum_sq = 0.0f;
  for (int i = threadIdx.x; i < H_vec; i += BLOCK_SIZE)
  {
    VecT v = x_vec[i];
    const T *v_as_T = reinterpret_cast<const T *>(&v);
#pragma unroll
    for (int k = 0; k < VEC; ++k)
    {
      float f = to_float<T>(v_as_T[k]);
      sum_sq += f * f;
    }
  }

  sum_sq = block_reduce_sum<BLOCK_SIZE>(sum_sq, smem);
  const float mean_sq = sum_sq / static_cast<float>(H);
  const float r = rsqrtf(mean_sq + eps);

  if (threadIdx.x == 0)
    rrms[row] = r;

  // Pass 2: normalize and scale
  VecT *y_vec = reinterpret_cast<VecT *>(y_row);
  const VecT *g_vec = reinterpret_cast<const VecT *>(g);

  for (int i = threadIdx.x; i < H_vec; i += BLOCK_SIZE)
  {
    VecT vx = x_vec[i];
    VecT vg = g_vec[i];
    VecT vy;

    const T *vx_T = reinterpret_cast<const T *>(&vx);
    const T *vg_T = reinterpret_cast<const T *>(&vg);
    T *vy_T = reinterpret_cast<T *>(&vy);

#pragma unroll
    for (int k = 0; k < VEC; ++k)
    {
      float fx = to_float<T>(vx_T[k]);
      float fg = to_float<T>(vg_T[k]);
      vy_T[k] = from_float<T>(fx * r * fg);
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
    const T *x_ptr, const T *g_ptr, T *y_ptr, float *rrms_ptr,
    int H, float eps)
{
  rmsnorm_forward_kernel<T, BLOCK_SIZE, VEC><<<grid, block, 0, stream>>>(
      x_ptr, g_ptr, y_ptr, rrms_ptr, H, eps);
}

template <typename T>
static void dispatch_rmsnorm_forward(
    dim3 grid, dim3 block, cudaStream_t stream,
    const T *x_ptr, const T *g_ptr, T *y_ptr, float *rrms_ptr,
    int H, float eps, int block_size, int vec)
{

  // VEC is fixed by T at compile time:
  //   float          -> VEC = 4  (float4 is the 16-byte vector)
  //   half/bfloat16  -> VEC = 8  (uint4 packs 8 half-words into 16 bytes)
  constexpr int VEC = (sizeof(T) == 4) ? 4 : 8;
  (void)vec; // vec is computed host-side and matches sizeof(T) by construction

  if (block_size == 1024)
    launch_rmsnorm_forward<T, 1024, VEC>(grid, block, stream, x_ptr, g_ptr, y_ptr, rrms_ptr, H, eps);
  else if (block_size == 512)
    launch_rmsnorm_forward<T, 512, VEC>(grid, block, stream, x_ptr, g_ptr, y_ptr, rrms_ptr, H, eps);
  else if (block_size == 256)
    launch_rmsnorm_forward<T, 256, VEC>(grid, block, stream, x_ptr, g_ptr, y_ptr, rrms_ptr, H, eps);
  else
    launch_rmsnorm_forward<T, 128, VEC>(grid, block, stream, x_ptr, g_ptr, y_ptr, rrms_ptr, H, eps);
}

// ===========================================================================
// BACKWARD
// ===========================================================================

// ---------------------------------------------------------------------------
// dx kernel: one block per row. Reuses rrms cached from forward.
//   dx_i = rrms*g_i*dy_i  -  (x_i * rrms^3 / H) * sum_j(x_j * g_j * dy_j)
// ---------------------------------------------------------------------------
template <typename T, int BLOCK_SIZE, int VEC>
__global__ void rmsnorm_backward_dx_kernel(
    const T* __restrict__ dy,
    const T* __restrict__ x,
    const T* __restrict__ g,
    const float* __restrict__ rrms,
    T* __restrict__ dx,
    int H) {

  using VecT = typename VecTypeHelper<T, VEC>::type;

  const int row = blockIdx.x;
  const T* dy_row = dy + row * H;
  const T* x_row  = x  + row * H;
  T*       dx_row = dx + row * H;
  const float r   = rrms[row];

  __shared__ float smem[BLOCK_SIZE / 32];

  const VecT* dy_vec = reinterpret_cast<const VecT*>(dy_row);
  const VecT* x_vec  = reinterpret_cast<const VecT*>(x_row);
  const VecT* g_vec  = reinterpret_cast<const VecT*>(g);
  const int H_vec = H / VEC;

  // Pass 1: compute the coupling sum  S = sum_j (x_j * g_j * dy_j), in fp32.
  float s = 0.0f;
  for (int i = threadIdx.x; i < H_vec; i += BLOCK_SIZE) {
    VecT vdy = dy_vec[i];
    VecT vx  = x_vec[i];
    VecT vg  = g_vec[i];
    const T* dy_T = reinterpret_cast<const T*>(&vdy);
    const T* x_T  = reinterpret_cast<const T*>(&vx);
    const T* g_T  = reinterpret_cast<const T*>(&vg);
    #pragma unroll
    for (int k = 0; k < VEC; ++k) {
      s += to_float<T>(x_T[k]) * to_float<T>(g_T[k]) * to_float<T>(dy_T[k]);
    }
  }
  s = block_reduce_sum<BLOCK_SIZE>(s, smem);

  // coefficient for the correction term
  const float coeff = (r * r * r) / static_cast<float>(H);

  // Pass 2: write dx.
  VecT* dx_vec = reinterpret_cast<VecT*>(dx_row);
  for (int i = threadIdx.x; i < H_vec; i += BLOCK_SIZE) {
    VecT vdy = dy_vec[i];
    VecT vx  = x_vec[i];
    VecT vg  = g_vec[i];
    VecT vdx;
    const T* dy_T = reinterpret_cast<const T*>(&vdy);
    const T* x_T  = reinterpret_cast<const T*>(&vx);
    const T* g_T  = reinterpret_cast<const T*>(&vg);
    T*       dx_T = reinterpret_cast<T*>(&vdx);
    #pragma unroll
    for (int k = 0; k < VEC; ++k) {
      float fx  = to_float<T>(x_T[k]);
      float fg  = to_float<T>(g_T[k]);
      float fdy = to_float<T>(dy_T[k]);
      float val = r * fg * fdy - coeff * fx * s;
      dx_T[k] = from_float<T>(val);
    }
    dx_vec[i] = vdx;
  }
}

// ---------------------------------------------------------------------------
// dg kernel: column reduction over rows.
//   dg_i = sum_over_rows ( x[row,i] * rrms[row] * dy[row,i] )
// 2D grid: blockIdx.x tiles columns, threads grid-stride down rows.
// Each block handles BLOCK_COLS columns and reduces all N rows for them,
// accumulating per-column partials in fp32. Block layout: threads are
// (TILE_COLS x ROWS_PER_BLOCK). Partial sums combined via shared memory.
// ---------------------------------------------------------------------------
template <typename T, int TILE_COLS, int ROWS_PER_BLOCK>
__global__ void rmsnorm_backward_dg_kernel(
    const T* __restrict__ dy,
    const T* __restrict__ x,
    const float* __restrict__ rrms,
    float* __restrict__ dg,   // fp32 accumulator, length H
    int N, int H) {

  // Column this thread-column is responsible for.
  const int col = blockIdx.x * TILE_COLS + threadIdx.x;

  // Shared tile: ROWS_PER_BLOCK partials per column, reduced at the end.
  __shared__ float smem[ROWS_PER_BLOCK][TILE_COLS];

  float acc = 0.0f;
  if (col < H) {
    // Grid-stride down the rows handled by this block's row-threads.
    for (int row = threadIdx.y; row < N; row += ROWS_PER_BLOCK) {
      float fx  = to_float<T>(x[row * H + col]);
      float fdy = to_float<T>(dy[row * H + col]);
      acc += fx * rrms[row] * fdy;
    }
  }
  smem[threadIdx.y][threadIdx.x] = acc;
  __syncthreads();

  // Reduce the ROWS_PER_BLOCK partials for each column (tree reduction over y).
  #pragma unroll
  for (int stride = ROWS_PER_BLOCK / 2; stride > 0; stride >>= 1) {
    if (threadIdx.y < stride) {
      smem[threadIdx.y][threadIdx.x] += smem[threadIdx.y + stride][threadIdx.x];
    }
    __syncthreads();
  }

  if (threadIdx.y == 0 && col < H) {
    dg[col] = smem[0][threadIdx.x];
  }
}

// ---------------------------------------------------------------------------
// dx launch helpers (same pattern as forward).
// ---------------------------------------------------------------------------
template <typename T, int BLOCK_SIZE, int VEC>
static inline void launch_dx(
    dim3 grid, dim3 block, cudaStream_t stream,
    const T* dy, const T* x, const T* g, const float* rrms, T* dx, int H) {
  rmsnorm_backward_dx_kernel<T, BLOCK_SIZE, VEC><<<grid, block, 0, stream>>>(
      dy, x, g, rrms, dx, H);
}

template <typename T>
static void dispatch_dx(
    dim3 grid, dim3 block, cudaStream_t stream,
    const T* dy, const T* x, const T* g, const float* rrms, T* dx,
    int H, int block_size) {
  constexpr int VEC = (sizeof(T) == 4) ? 4 : 8;
  if      (block_size == 1024) launch_dx<T, 1024, VEC>(grid, block, stream, dy, x, g, rrms, dx, H);
  else if (block_size == 512)  launch_dx<T, 512,  VEC>(grid, block, stream, dy, x, g, rrms, dx, H);
  else if (block_size == 256)  launch_dx<T, 256,  VEC>(grid, block, stream, dy, x, g, rrms, dx, H);
  else                          launch_dx<T, 128,  VEC>(grid, block, stream, dy, x, g, rrms, dx, H);
}

// ---------------------------------------------------------------------------
// Host entry point for backward.
// ---------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> rmsnorm_backward_cuda(
    torch::Tensor dy,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor rrms) {

  const int H = x.size(-1);
  const int N = x.numel() / H;

  auto dx = torch::empty_like(x);
  // dg accumulates in fp32 then casts back to weight dtype at the end.
  auto dg_f32 = torch::zeros({H}, x.options().dtype(torch::kFloat32));

  const int vec = (x.element_size() == 4) ? 4 : 8;
  TORCH_CHECK(H % vec == 0, "H must be divisible by ", vec);

  int block_size = 1024;
  while (block_size > 32 && block_size > (H / vec)) block_size >>= 1;

  auto stream = at::cuda::getCurrentCUDAStream();

  // ---- dx ----
  {
    const dim3 grid(N);
    const dim3 block(block_size);
    AT_DISPATCH_SWITCH(x.scalar_type(), "rmsnorm_backward_dx",
      AT_DISPATCH_CASE(at::ScalarType::Float, [&] {
        using T = float;
        dispatch_dx<T>(grid, block, stream,
            reinterpret_cast<const T*>(dy.data_ptr<float>()),
            reinterpret_cast<const T*>(x.data_ptr<float>()),
            reinterpret_cast<const T*>(weight.data_ptr<float>()),
            rrms.data_ptr<float>(),
            reinterpret_cast<T*>(dx.data_ptr<float>()),
            H, block_size);
      })
      AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
        using T = __half;
        dispatch_dx<T>(grid, block, stream,
            reinterpret_cast<const T*>(dy.data_ptr<at::Half>()),
            reinterpret_cast<const T*>(x.data_ptr<at::Half>()),
            reinterpret_cast<const T*>(weight.data_ptr<at::Half>()),
            rrms.data_ptr<float>(),
            reinterpret_cast<T*>(dx.data_ptr<at::Half>()),
            H, block_size);
      })
      AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
        using T = __nv_bfloat16;
        dispatch_dx<T>(grid, block, stream,
            reinterpret_cast<const T*>(dy.data_ptr<at::BFloat16>()),
            reinterpret_cast<const T*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<const T*>(weight.data_ptr<at::BFloat16>()),
            rrms.data_ptr<float>(),
            reinterpret_cast<T*>(dx.data_ptr<at::BFloat16>()),
            H, block_size);
      })
    );
  }

  // ---- dg ----
  {
    constexpr int TILE_COLS = 32;
    constexpr int ROWS_PER_BLOCK = 16;
    const dim3 grid((H + TILE_COLS - 1) / TILE_COLS);
    const dim3 block(TILE_COLS, ROWS_PER_BLOCK);
    AT_DISPATCH_SWITCH(x.scalar_type(), "rmsnorm_backward_dg",
      AT_DISPATCH_CASE(at::ScalarType::Float, [&] {
        using T = float;
        rmsnorm_backward_dg_kernel<T, TILE_COLS, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(
            reinterpret_cast<const T*>(dy.data_ptr<float>()),
            reinterpret_cast<const T*>(x.data_ptr<float>()),
            rrms.data_ptr<float>(), dg_f32.data_ptr<float>(), N, H);
      })
      AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
        using T = __half;
        rmsnorm_backward_dg_kernel<T, TILE_COLS, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(
            reinterpret_cast<const T*>(dy.data_ptr<at::Half>()),
            reinterpret_cast<const T*>(x.data_ptr<at::Half>()),
            rrms.data_ptr<float>(), dg_f32.data_ptr<float>(), N, H);
      })
      AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
        using T = __nv_bfloat16;
        rmsnorm_backward_dg_kernel<T, TILE_COLS, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(
            reinterpret_cast<const T*>(dy.data_ptr<at::BFloat16>()),
            reinterpret_cast<const T*>(x.data_ptr<at::BFloat16>()),
            rrms.data_ptr<float>(), dg_f32.data_ptr<float>(), N, H);
      })
    );
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto dg = dg_f32.to(weight.scalar_type());
  return {dx, dg};
}

// ---------------------------------------------------------------------------
// Host entry point.
// ---------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> rmsnorm_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    double eps)
{

  const int H = x.size(-1);
  const int N = x.numel() / H;

  auto y = torch::empty_like(x);
  auto rrms = torch::empty({N}, x.options().dtype(torch::kFloat32));

  const int vec = (x.element_size() == 4) ? 4 : 8;
  TORCH_CHECK(H % vec == 0,
              "H must be divisible by ", vec, " (got ", H, ")");

  int block_size = 1024;
  while (block_size > 32 && block_size > (H / vec))
    block_size >>= 1;

  const dim3 grid(N);
  const dim3 block(block_size);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_SWITCH(x.scalar_type(), "rmsnorm_forward_cuda",
                     AT_DISPATCH_CASE(at::ScalarType::Float, [&]
                                      {
      using T = float;
      dispatch_rmsnorm_forward<T>(
          grid, block, stream,
          reinterpret_cast<const T*>(x.data_ptr<float>()),
          reinterpret_cast<const T*>(weight.data_ptr<float>()),
          reinterpret_cast<T*>(y.data_ptr<float>()),
          rrms.data_ptr<float>(),
          H, static_cast<float>(eps), block_size, vec); })
                         AT_DISPATCH_CASE(at::ScalarType::Half, [&]
                                          {
      using T = __half;
      dispatch_rmsnorm_forward<T>(
          grid, block, stream,
          reinterpret_cast<const T*>(x.data_ptr<at::Half>()),
          reinterpret_cast<const T*>(weight.data_ptr<at::Half>()),
          reinterpret_cast<T*>(y.data_ptr<at::Half>()),
          rrms.data_ptr<float>(),
          H, static_cast<float>(eps), block_size, vec); })
                             AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&]
                                              {
      using T = __nv_bfloat16;
      dispatch_rmsnorm_forward<T>(
          grid, block, stream,
          reinterpret_cast<const T*>(x.data_ptr<at::BFloat16>()),
          reinterpret_cast<const T*>(weight.data_ptr<at::BFloat16>()),
          reinterpret_cast<T*>(y.data_ptr<at::BFloat16>()),
          rrms.data_ptr<float>(),
          H, static_cast<float>(eps), block_size, vec); }));

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, rrms};
}