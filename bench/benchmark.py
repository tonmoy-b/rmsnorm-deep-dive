import torch
import torch.nn.functional as F

import csv
import time

import rmsnorm
from rmsnorm.triton_rmsnorm import rmsnorm_forward_triton

"""Benchmark RMSNorm: custom CUDA vs Triton vs PyTorch eager (F.rms_norm).
Reports latency (us) and achieved memory bandwidth (GB/s). 
Flops not a key factor here as RMSNorm is memory-bound, due to which bandwidth utilization is examined.
"""

def benchmark_fn(fn, *args, warmup=25, iters=100):
    """Time a GPU function with CUDA events. Returns mean latency in microseconds."""
    # Warmup. This is also meant to trigger Triton JIT/cuDNN autotune with first calls.
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    return (total_ms / iters) * 1000.0  # ms -> us


def bandwidth_gbps(n, h, dtype, latency_us):
    """Effective bandwidth: read x + write y = 2*N*H*sizeof bytes."""
    bytes_moved = 2 * n * h * torch.tensor([], dtype=dtype).element_size()
    seconds = latency_us * 1e-6
    return (bytes_moved / seconds) / 1e9


def main_without_datacollection():
    assert torch.cuda.is_available(), "need CUDA"
    dev = torch.cuda.get_device_name(0)
    print(f"Device: {dev}")
    print(f"{'shape':>18} {'dtype':>9} {'impl':>8} {'lat(us)':>9} {'BW(GB/s)':>9}")
    print("-" * 60)
    #define the shapes
    shapes = [(1024, 1024), (2048, 4096), (4096, 4096), (8192, 4096), (1024, 8192)]
    dtypes = [torch.float16, torch.bfloat16, torch.float32]
    eps = 1e-6 #error range that is ok
    for dtype in dtypes:
        for (N, H) in shapes:
            x = torch.randn(N, H, device="cuda", dtype=dtype)
            g = torch.randn(H, device="cuda", dtype=dtype)

            impls = {
                "eager":  lambda: F.rms_norm(x, (H,), g, eps),
                "cuda":   lambda: rmsnorm.forward(x, g, eps),
                "triton": lambda: rmsnorm_forward_triton(x, g, eps),
            }

            for name, fn in impls.items():
                try:
                    lat = benchmark_fn(fn)
                    bw = bandwidth_gbps(N, H, dtype, lat)
                    print(f"{str((N,H)):>18} {str(dtype).replace('torch.',''):>9} "
                          f"{name:>8} {lat:>9.2f} {bw:>9.1f}")
                except Exception as e:
                    print(f"{str((N,H)):>18} {str(dtype).replace('torch.',''):>9} "
                          f"{name:>8} FAILED: {type(e).__name__}: {e}")
            print()


def main():
    assert torch.cuda.is_available(), "need CUDA"
    dev = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"Device: {dev} (sm_{cap[0]}{cap[1]})")
    print(f"{'shape':>18} {'dtype':>9} {'impl':>8} {'lat(us)':>9} {'BW(GB/s)':>9}")
    print("-" * 60)

    shapes = [(1024, 1024), (2048, 4096), (4096, 4096), (8192, 4096), (1024, 8192)]
    dtypes = [torch.float16, torch.bfloat16, torch.float32]
    eps = 1e-6

    rows = []  # collected for CSV

    for dtype in dtypes:
        for (N, H) in shapes:
            x = torch.randn(N, H, device="cuda", dtype=dtype)
            g = torch.randn(H, device="cuda", dtype=dtype)

            impls = {
                "eager":  lambda: F.rms_norm(x, (H,), g, eps),
                "cuda":   lambda: rmsnorm.forward(x, g, eps),
                "triton": lambda: rmsnorm_forward_triton(x, g, eps),
            }

            for name, fn in impls.items():
                try:
                    lat = benchmark_fn(fn)
                    bw = bandwidth_gbps(N, H, dtype, lat)
                    dtype_str = str(dtype).replace("torch.", "")
                    print(f"{str((N,H)):>18} {dtype_str:>9} "
                          f"{name:>8} {lat:>9.2f} {bw:>9.1f}")
                    rows.append({
                        "device": dev,
                        "compute_capability": f"sm_{cap[0]}{cap[1]}",
                        "N": N,
                        "H": H,
                        "dtype": dtype_str,
                        "impl": name,
                        "latency_us": round(lat, 3),
                        "bandwidth_gbps": round(bw, 2),
                    })
                except Exception as e:
                    print(f"{str((N,H)):>18} {str(dtype).replace('torch.',''):>9} "
                          f"{name:>8} FAILED: {type(e).__name__}: {e}")
            print()

    # Write CSV with a timestamp so reruns don't overwrite each other.
    import os
    os.makedirs("bench/results", exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"bench/results/rmsnorm_bench_{stamp}.csv"
    fieldnames = ["device", "compute_capability", "N", "H", "dtype",
                  "impl", "latency_us", "bandwidth_gbps"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()