"""Cost of a non-foldable channel scale: NVFP4 activation quantization with and without a fused per-channel x / s,
vs a fused 16x16 block Hadamard (torch.compile'd kernels, CUDA events, median of 100). Blackwell sm_120."""
import json, sys, torch, torch._dynamo
torch._dynamo.config.cache_size_limit = 256
if hasattr(torch._dynamo.config, "recompile_limit"): torch._dynamo.config.recompile_limit = 256
dev = "cuda"
E2 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6.], device=dev); TH = (E2[1:] + E2[:-1]) / 2
H = torch.ones(1, 1, device=dev)
while H.shape[0] < 16: H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
H16 = (H / 4).to(torch.bfloat16)


def q(x, inv_s=None, rot=False):
    M, K = x.shape
    if inv_s is not None: x = x * inv_s
    if rot: x = (x.view(M, K // 16, 16) @ H16).view(M, K)
    xb = x.float().view(M, K // 16, 16); amax = xb.abs().amax(-1, keepdim=True).clamp(min=1e-12)
    s = (amax / 6.0).to(torch.float8_e4m3fn).float()
    idx = torch.bucketize((xb.abs() / s).clamp(max=6.0), TH)
    code = (idx | ((xb < 0).to(torch.int64) << 3)).to(torch.uint8).view(M, K)
    return (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous(), s.to(torch.float8_e4m3fn)


qp = torch.compile(lambda x: q(x), dynamic=False)
qs = torch.compile(lambda x, i: q(x, i), dynamic=False)
qr = torch.compile(lambda x: q(x, rot=True), dynamic=False)


def bench(fn, n=100):
    for _ in range(10): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    ts.sort(); return ts[n // 2]


rows = []
for K in [4096, 14336]:
    for M in [1, 128, 2048, 8192]:
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16); inv = (torch.rand(K, device=dev) + 0.5).to(torch.bfloat16)
        r = dict(M=M, K=K, plain=bench(lambda: qp(x)), scaled=bench(lambda: qs(x, inv)), rot=bench(lambda: qr(x)))
        rows.append(r); print(f"K={K:5d} M={M:5d} plain {r['plain']*1e3:7.1f} us  +x/s {r['scaled']*1e3:7.1f} us ({100*(r['scaled']/r['plain']-1):+5.1f}%)  +H16 {r['rot']*1e3:7.1f} us ({100*(r['rot']/r['plain']-1):+5.1f}%)", flush=True)
json.dump(dict(device=torch.cuda.get_device_name(0), rows=rows), open(sys.argv[1], "w"), indent=1)
