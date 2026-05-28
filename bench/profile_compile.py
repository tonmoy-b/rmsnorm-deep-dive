import torch
import torch.nn as nn
import torch.nn.functional as F

import rmsnorm

""" torch.compile / TorchInductor analysis.
Compare a module built on our custom CUDA op against one built on eager F.rms_norm, under torch.compile. 
Examination Points:
  - graph breaks 
  - fusion 
  - dispatcher overhead 
"""

# Define 2 pytorch modules with the same math but the different RMSNorm implementations for comparison
# For Each ==> rmsnorm -> elementwise (gelu) -> rmsnorm. 

class CustomOpBlock(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.g1 = nn.Parameter(torch.ones(H))
        self.g2 = nn.Parameter(torch.ones(H))

    def forward(self, x):
        x = rmsnorm.rmsnorm(x, self.g1)      # custom autograd.Function -> C++
        x = F.gelu(x)
        x = rmsnorm.rmsnorm(x, self.g2)
        return x


class EagerBlock(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.g1 = nn.Parameter(torch.ones(H))
        self.g2 = nn.Parameter(torch.ones(H))

    def forward(self, x):
        x = F.rms_norm(x, (x.shape[-1],), self.g1)  # native ATen op
        x = F.gelu(x)
        x = F.rms_norm(x, (x.shape[-1],), self.g2)
        return x


def count_graph_breaks(mod, x, label):
    """Use dynamo.explain to report graph count and breaks."""
    torch._dynamo.reset()
    explanation = torch._dynamo.explain(mod)(x)
    print(f"\n=== {label}: dynamo.explain ===")
    print(f"  graph count:        {explanation.graph_count}")
    print(f"  graph break count:  {explanation.graph_break_count}")
    print(f"  op count:           {explanation.op_count}")
    if explanation.break_reasons:
        print("  break reasons:")
        for i, r in enumerate(explanation.break_reasons):
            reason = getattr(r, "reason", str(r))
            print(f"    [{i}] {reason}")


def time_module(mod, x, warmup=25, iters=100):
    for _ in range(warmup):
        mod(x)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        mod(x)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000.0  # us


def main():
    assert torch.cuda.is_available()
    H = 4096
    N = 4096
    x = torch.randn(N, H, device="cuda", dtype=torch.float16)

    custom = CustomOpBlock(H).cuda().half()
    eager  = EagerBlock(H).cuda().half()

    # Graph break analysis 
    count_graph_breaks(custom, x, "CustomOpBlock")
    count_graph_breaks(eager,  x, "EagerBlock")

    # Compile both custom & eager setups
    custom_c = torch.compile(custom)
    eager_c  = torch.compile(eager)

    # Trigger the compilation ~ first call compiles here
    custom_c(x); eager_c(x)
    torch.cuda.synchronize()

    # Latency Timings: eager module vs compiled, both implementations seen 
    print("\n=== Latency (us), N={}, H={}, fp16 ===".format(N, H))
    print(f"  custom  eager-mode : {time_module(custom,   x):8.2f}")
    print(f"  custom  compiled   : {time_module(custom_c, x):8.2f}")
    print(f"  eager   eager-mode : {time_module(eager,    x):8.2f}")
    print(f"  eager   compiled   : {time_module(eager_c,  x):8.2f}")

    # Profiler: op breakdown / dispatcher overhead 
    from torch.profiler import profile, ProfilerActivity
    print("\n=== Profiler: EagerBlock compiled (top ops by CUDA time) ===")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(20):
            eager_c(x)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print("\n=== Profiler: CustomOpBlock compiled (top ops by CUDA time) ===")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(20):
            custom_c(x)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


if __name__ == "__main__":
    main()