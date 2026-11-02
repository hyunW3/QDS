"""GEMM latency on Blackwell (sm_120): BF16 vs FP8 (E4M3, per-tensor) vs NVFP4 (E2M1 + E4M3 1x16 block scales).

For each model's decoder-layer Linear shapes and token counts M, times the GEMM alone (CUDA events, median of 50)
and, separately, the cost of quantizing the activation to NVFP4 with a torch.compile'd kernel (upper bound: real
deployments fuse this into the producer op), plus the extra cost of 4/6 scale selection and of a 16x16 RHT.
Weights are pre-quantized (static). Values are random; only timing is measured.
"""
import json, sys, time
import torch

dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False
MODELS = {  # (K, N) per Linear in one decoder layer; qkv fused, gate/up fused
    "1.3B-class (d2048, ffn5632)": [(2048, 3 * 2048), (2048, 2048), (2048, 2 * 5632), (5632, 2048)],
    "Qwen2.5-1.5B (d1536, ffn8960, GQA)": [(1536, 1536 + 2 * 256), (1536, 1536), (1536, 2 * 8960), (8960, 1536)],
    "Llama-3-8B (d4096, ffn14336, GQA)": [(4096, 4096 + 2 * 1024), (4096, 4096), (4096, 2 * 14336), (14336, 4096)],
}
MS = [1, 8, 32, 128, 512, 2048, 8192]


def bench(fn, n=50, warm=10):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts) // 2]


E2 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6.], device=dev)


def nvfp4_quant(x, four_six=False):
    """x [M,K] bf16 -> (packed e2m1 [M,K/2] uint8, e4m3 scales [M*K/16]). Codes: sign bit 3, magnitude index 0..7."""
    M, K = x.shape
    xb = x.float().view(M, K // 16, 16)
    amax = xb.abs().amax(-1, keepdim=True).clamp(min=1e-12)
    th = (E2[1:] + E2[:-1]) / 2

    def enc(s):
        t = (xb.abs() / s).clamp(max=6.0)
        idx = torch.bucketize(t, th)
        return idx, (E2[idx] * s - xb.abs()).pow(2).sum(-1, keepdim=True)
    s = (amax / 6.0).to(torch.float8_e4m3fn).float()
    idx, err = enc(s)
    if four_six:
        s4 = (amax / 4.0).to(torch.float8_e4m3fn).float()
        idx4, err4 = enc(s4)
        pick = err4 < err
        idx = torch.where(pick, idx4, idx); s = torch.where(pick, s4, s)
    code = (idx | ((xb < 0).to(torch.int64) << 3)).to(torch.uint8).view(M, K)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    return packed, s.to(torch.float8_e4m3fn).reshape(-1)


H16 = None


def rht(x):
    global H16
    if H16 is None:
        H = torch.ones(1, 1, device=dev)
        while H.shape[0] < 16:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        H16 = (H / 4).to(torch.bfloat16)
    M, K = x.shape
    return (x.view(M, K // 16, 16) @ H16).view(M, K)


import torch._dynamo
torch._dynamo.config.cache_size_limit = 1024          # one specialization per (M, K); default 8 silently falls back to eager
torch._dynamo.config.accumulated_cache_size_limit = 8192
if hasattr(torch._dynamo.config, "recompile_limit"):
    torch._dynamo.config.recompile_limit = 1024
q_plain = torch.compile(lambda x: nvfp4_quant(x, False), dynamic=False)
q_46 = torch.compile(lambda x: nvfp4_quant(x, True), dynamic=False)
q_rht = torch.compile(lambda x: nvfp4_quant(rht(x), False), dynamic=False)

res = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "rows": []}
print(res["device"], torch.__version__, flush=True)
one = torch.ones((), device=dev)
for mname, shapes in MODELS.items():
    for M in MS:
        tot = {k: 0.0 for k in ["bf16", "fp8", "nvfp4", "q_plain", "q_46", "q_rht"]}
        for K, N in shapes:
            a = torch.randn(M, K, device=dev, dtype=torch.bfloat16); w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
            tot["bf16"] += bench(lambda: a @ w.t())
            a8, w8 = a.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)
            tot["fp8"] += bench(lambda: torch._scaled_mm(a8, w8.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16))
            Mp = max(M, 128)                                    # block-scaled kernels want padded rows
            a4 = torch.randint(0, 255, (Mp, K // 2), device=dev, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
            w4 = torch.randint(0, 255, (N, K // 2), device=dev, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
            sa = torch.ones(Mp * K // 16, device=dev).to(torch.float8_e4m3fn); sb = torch.ones(N * K // 16, device=dev).to(torch.float8_e4m3fn)
            tot["nvfp4"] += bench(lambda: torch._scaled_mm(a4, w4.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16))
            tot["q_plain"] += bench(lambda: q_plain(a)); tot["q_46"] += bench(lambda: q_46(a)); tot["q_rht"] += bench(lambda: q_rht(a))
        row = dict(model=mname, M=M, **{k: round(v, 4) for k, v in tot.items()})
        res["rows"].append(row)
        print(f"{mname:36s} M={M:5d} | bf16 {tot['bf16']:.3f} ms  fp8 {tot['fp8']:.3f} ({tot['bf16']/tot['fp8']:.2f}x)  nvfp4 {tot['nvfp4']:.3f} ({tot['bf16']/tot['nvfp4']:.2f}x) | act-quant plain {tot['q_plain']:.3f}  4/6 {tot['q_46']:.3f}  rht {tot['q_rht']:.3f} ms", flush=True)
json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "gemm_bench.json", "w"), indent=1)
