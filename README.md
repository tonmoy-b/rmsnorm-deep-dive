# RMSNorm: A Deep Dive in CUDA, Triton, and `torch.compile`

This Repo has three implementations of RMSNorm — a hand-written CUDA kernel, an OpenAI Triton kernel, and a PyTorch eager. Furthermore, these have been benchmarked head-to-head and analyzed under `torch.compile`/TorchInductor, with a focus on graph breaks, fusion, and dispatcher overhead.

The goal of this project is to demonstrate the full loop of authoring a custom neural-network operator: writing the kernel, binding it to PyTorch with correct autograd, making it numerically robust across dtypes, benchmarking it accurately against the framework's own fused kernel, and understanding exactly how it behaves inside the compiler stack. The goal of this project is not to beat PyTorch.

## Quick start
This is a CUDA extension that must be built before use. Build with 
>cmake -G Ninja -B build && cmake --build build 
(see Build and run for full requirements). The build produces rmsnorm/_C.<python-tag>.pyd, which is gitignored. This is to conform with the standard practice of not commiting build artifacts and also since every clone must rebuild for its local Python/CUDA/torch combination.

## Headline results

- **Custom CUDA and Triton both reach ~167 GB/s on an RTX 3050 Laptop GPU — about 87% of the card's ~192 GB/s peak memory bandwidth.** RMSNorm is memory-bound, so bandwidth utilization, not FLOPs, is the metric that matters.
- **In isolation, the custom kernels are ~1.4× faster than PyTorch's `F.rms_norm`** at LLM-typical shapes (steady-state, fp16/bf16).
- **Inside `torch.compile`, that result inverts.** A module built on `F.rms_norm` fuses RMSNorm + GeLU into a single Triton kernel and runs **1.7× faster than the same module built on the custom op**, because a slower kernel that fuses beats a faster kernel that does not.
- **A raw pybind11 custom op causes graph breaks** (3 graphs / 2 breaks for a `norm → gelu → norm` block). Registering it via `torch.library.custom_op` with a fake/meta implementation **eliminates the breaks (1 graph / 0 breaks)** and turns it into a first-class dispatcher op.
 

## The operator

For input `x` of shape `(…, H)` and a learnable weight `g` of shape `(H,)`:

$$y = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot g$$

```
rms(x) = sqrt(mean(x²) + eps)      # reduced over the last dimension
y      = (x / rms(x)) · g
```
**Why separate the normalization and the weight?**
* **The Division (RMS):** Stabilizes training by forcing the activations into a predictable numeric range, which prevents massive gradient spikes.
* **The Multiplication ($g$):** Restores the network's expressive power. Because pure normalization destroys the original magnitude of the features, $g$ acts as a set of per-channel "volume knobs." During training, the optimizer learns to dynamically scale up critical feature dimensions and dial down noisy ones.

RMSNorm is used in essentially every modern LLM (Llama, Mistral, Qwen). It is a pure reduction-plus-elementwise op with no matmul, which makes it a clean, recognizable target for a first custom kernel and a textbook memory-bound workload.


All implementations accumulate the sum of squares in **fp32 regardless of
storage dtype**. This is why the fp16/bf16 forward output is bit-identical to
`F.rms_norm`: both round exactly once, at the end.

## Three implementations

### 1. CUDA (`csrc/rmsnorm_cuda.cu`)

One thread block per row. Within a block:

- 128-bit vectorized loads (`float4` for fp32, `uint4` packing 8 halves for fp16/bf16) to saturate memory transactions.
- Sum of squares via a warp-shuffle reduction (`__shfl_xor_sync`) followed by a shared-memory cross-warp reduction.
- fp32 accumulation, cast back to storage dtype only on the final write.
- Backward as two kernels: `dx` reuses the per-row reduction (the gradient has a coupling term, `Σⱼ xⱼ·gⱼ·dyⱼ`), and `dg` is a separate tiled column reduction over the batch dimension with fp32 accumulation and no atomics.

### 2. Triton (`rmsnorm/triton_rmsnorm.py`)

The same forward in ~25 lines instead of ~100. The whole row is loaded as one
`BLOCK_H`-wide vector; `tl.sum` replaces the entire hand-written reduction; the
compiler maps the logical block onto hardware threads. Masked loads handle
non-power-of-two `H` for free — a shape the CUDA kernel rejects by design.

This is the core philosophical contrast: **in CUDA you manage threads; in Triton
you manage blocks of data and the compiler manages threads.** You trade
fine-grained control for a large reduction in code and a compiler that often
matches hand-tuned performance.

### 3. Eager

`torch.nn.functional.rms_norm`, PyTorch's own fused kernel, used as the
correctness reference and the performance baseline.

## Benchmarks

Measured with CUDA events (25 warmup, 100 timed iterations). Effective bandwidth
is `2·N·H·sizeof(dtype) / latency` (read `x`, write `y`). Full data in
`bench/results/`.

Representative fp16 results (RTX 3050 Laptop GPU, sm_86):

| Shape (N, H) | Impl | Latency (µs) | Bandwidth (GB/s) |
|---|---|---|---|
| (4096, 4096) | eager | 588.8 | 114.0 |
| (4096, 4096) | cuda | 407.3 | 164.8 |
| (4096, 4096) | triton | 405.4 | 165.6 |
| (8192, 4096) | eager | 1174.7 | 114.3 |
| (8192, 4096) | cuda | 806.9 | 166.3 |
| (8192, 4096) | triton | 802.1 | 167.3 |

### How to read this

- **Latency and bandwidth are the same number expressed two ways.** For a fixed shape and dtype, `bandwidth = constant / latency`. Bandwidth is the normalized metric: it divides out problem size so different shapes can be compared on efficiency.
- **Bandwidth rises with problem size, then plateaus** (~167 GB/s). At small shapes, fixed kernel-launch overhead forms a large fraction of runtime. However, as the problem grows it amortizes and the kernel converges to a true steady-state efficiency. That plateau — ~87% of the card's ~192 GB/s peak — is the real result.
- **CUDA and Triton converge** because both are memory-bound and both sit near the bandwidth ceiling. When you are already at ~87% of peak, there is no room for one implementation to beat another; the HBM is the bottleneck, not the code. The one regime where they diverge is very small shapes, where Triton's higher per-launch overhead makes it noticeably slower.
- **fp16 ≈ bf16** (identical byte width → identical memory traffic). **fp32 is ~2× slower** (double the bytes) but reaches the *same* bandwidth — confirming the kernels are bandwidth-bound and dtype-agnostic in efficiency.

### A note on hardware limitations

Recorded statistics are laptop-GPU numbers, measured under Windows WDDM with thermal throttling and other protective measures, this places limitations on performance. However, the **relative** comparisons (cuda vs triton vs eager, fp16 vs fp32) are valid regardless since they are all performed on the same platform. The **absolute** bandwidth (~167 GB/s) is specific to this card and would be likely be an order of magnitude higher on a datacenter GPU (perhaps on the order of an A100 at ~1.5–2 TB/s). The benchmark script is hardware-agnostic and writes a self-describing CSV, so the same study reruns unchanged on a cloud GPU. 

## `torch.compile` analysis

This is an involved part of the project and very relevant to PyTorch-internals work. The harness script (`bench/profile_compile.py`) wraps two
modules — one built on the custom op, one on `F.rms_norm` — each computing `rmsnorm → gelu → rmsnorm`, and compares them under `torch.compile`.

### Graph Breaks : Problem Encountered and Mitigated

### The Issue
A raw pybind11 custom op is opaque to TorchDynamo, which actually traces Python bytecode and thus can't see into compiled C++:

```
CustomOpBlock (raw pybind):  graph count: 3,  graph break count: 2
EagerBlock   (F.rms_norm):   graph count: 1,  graph break count: 0
```

Dynamo's own diagnostic names the cause exactly: *"Dynamo does not know how to trace the builtin `rmsnorm._C…forward` ... a third-party C/C++ Python extension (perhaps created with pybind)."* Each custom call splits the block into a separate graph.

### The Fix: register the op with `torch.library`

Wrapping the kernels in `torch.library.custom_op` with:

- a **`register_fake`** (meta) implementation that propagates shapes/dtypes without running the C++, so Dynamo can trace through it on FakeTensors, and
- a **`register_autograd`** hook so it stays differentiable under compile, turns the op into a first-class dispatcher node:

```
CustomOpBlock (registered):  graph count: 1,  graph break count: 0
```

The op now appears in the profiler as `rmsnorm::forward` rather than an untraceable pybind call.

### Fusion: the counterintuitive result

| Module | Eager (µs) | Compiled (µs) |
|---|---|---|
| Custom op | 1222 | 1222 |
| `F.rms_norm` | 1577 | **703** |

`torch.compile` gives the eager module a **2.2× speedup** by fusing both norms and the GeLU into a single kernel (`triton_red_fused__fused_rms_norm_gelu_0` — 100% of CUDA time). 
However, it does **nothing** for the custom op: the GeLU stays a separate kernel (`triton_poi_fused_gelu_0`), because Inductor treats the custom CUDA kernel as a black box it cannot fuse *into*.

The result: a module using PyTorch's *slower-in-isolation* `F.rms_norm` is **1.7× faster** than one using the custom *faster-in-isolation* kernel, once compiled. **A slower kernel that fuses beats a faster kernel that does not.**

### Two precise conclusions

1. **Registration fixes traceability, not fusability.** Registering the op eliminates graph breaks and recovers CPU-side dispatcher overhead, but Inductor still cannot fuse surrounding elementwise ops into an opaque kernel. Genuine fusion requires the op to be expressible in Inductor's IR (i.e. as a native/decomposable op or a Triton template), not just registered.
2. **Removing the graph breaks did not change wall-clock at this shape**, because the workload is GPU-bound and the break overhead was CPU-side dispatch latency hidden behind kernel execution. The benefit would surface in dispatch-bound regimes — many small ops, or CPU-limited inference — not here.

## Relevance to non-CUDA / custom accelerators

The pattern this project exercises — author a kernel, bind it through the dispatcher, give it a fake/meta implementation, register autograd, make it
visible to Dynamo/Inductor — is exactly the abstraction one extends to bring a new backend (a non-CUDA accelerator) into PyTorch. The fake implementation is backend-independent shape logic; the real implementation is the backend kernel; the dispatcher routes by device key. Porting this op to a different accelerator would mean swapping the real implementation and registering a new device key, leaving the fake/meta and autograd registration untouched.

## Build and run

Requires an NVIDIA GPU, CUDA Toolkit, PyTorch with matching CUDA, and a host C++ compiler. Developed and tested on Windows (VS 2026 / MSVC 14.50, CUDA 13.2, PyTorch 2.12 cu130, Python 3.12) against an RTX 3050 Laptop GPU (sm_86).

```bash
# Configure and build the CUDA extension (CMake + Ninja)
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Run correctness tests (forward, backward, triton)
python -m pytest tests/ -v -s

# Benchmark all three implementations -> bench/results/*.csv
python -m bench.benchmark

# torch.compile / graph-break / fusion analysis
python -m bench.profile_compile
```

The build links `torch_python` explicitly and forces torch discovery through the active interpreter; see `CMakeLists.txt` for the rationale on each flag.

## Correctness

All implementations are verified against an fp32-accumulation reference:

- **Forward**: fp32/fp16/bf16 across multiple shapes; fp16/bf16 are bit-identical to `F.rms_norm`.
- **Backward**: `dx` and `dg` compared against PyTorch autograd through the reference, in fp32 (atol/rtol 1e-4). `gradcheck` is intentionally not used — it requires fp64, which the kernels do not support, since no LLM uses fp64 for this op.

## Known limitations / future work

- **`dg` column-reduction kernel** walks all `N` rows per block (1-D grid over columns). Correct and fast for `N` up to a few thousand. But very large `N` would need, or at least benefit from, a 2-D grid plus a second-stage reduction.
- **Triton kernel** assumes a row fits in one program's registers (fine to `H ≈ 8–16k`). Larger `H` would need row tiling with a two-stage reduction.
- **No cross-op fusion** for the custom op under Inductor (see analysis above). Expressing RMSNorm as a Triton template registered with Inductor would enable it.
- **Triton backward** is not implemented; the Triton path is forward-only by design, with the CUDA path providing the full differentiable op.

## Repository layout

```
csrc/                  CUDA kernels + pybind11 binding
  rmsnorm.h
  rmsnorm.cpp          validation + dispatch + PYBIND11_MODULE
  rmsnorm_cuda.cu      forward + backward kernels
rmsnorm/               Python package
  __init__.py          torch.library custom-op registration + public API
  triton_rmsnorm.py    Triton forward kernel
tests/                 correctness tests (forward / backward / triton)
bench/
  benchmark.py         latency + bandwidth, writes CSV
  profile_compile.py   torch.compile graph-break / fusion / profiler analysis
  results/             benchmark CSVs
CMakeLists.txt
```
