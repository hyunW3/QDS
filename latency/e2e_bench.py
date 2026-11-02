"""End-to-end Llama-style decoder latency on Blackwell (sm_120): BF16 vs FP8 vs NVFP4 (E2M1) Linear layers.

Pure-torch decoder (RMSNorm, GQA SDPA attention with KV cache, SwiGLU), random weights (timing only).
Every forward is captured in a CUDA graph (serving-like, no CPU launch overhead), timed with CUDA events.
Linear modes (all decoder-layer Linears; embedding / lm_head stay BF16):
  bf16        F.linear
  fp8         dynamic per-tensor E4M3 activation quant + _scaled_mm (static E4M3 weights)
  nvfp4       dynamic NVFP4 activation quant (1x16 absmax, E4M3 scales; torch.compile'd, unfused) + _scaled_mm
  nvfp4_46    + 4/6 per-block scale selection
  nvfp4_rht   + 16x16 Hadamard on activations before quant (weights assumed pre-rotated)
  nvfp4_noq   GEMM only (activation already FP4: ideal fused producer), lower bound
Scenarios: prefill (B=1, T tokens, causal) and one decode step (B sequences, context C).
"""
import json, math, sys
import torch, torch.nn.functional as F
import torch._dynamo
torch._dynamo.config.cache_size_limit = 1024
torch._dynamo.config.accumulated_cache_size_limit = 8192
if hasattr(torch._dynamo.config, "recompile_limit"):
    torch._dynamo.config.recompile_limit = 1024
dev = "cuda"
torch.manual_seed(0)

CFGS = {  # d, ffn, n_layers, n_heads, n_kv, vocab
    "1.3B-class": dict(d=2048, ffn=5632, L=24, H=16, KV=16, V=32000),
    "Qwen2.5-1.5B-shape": dict(d=1536, ffn=8960, L=28, H=12, KV=2, V=151936),
    "Llama-3-8B-shape": dict(d=4096, ffn=14336, L=32, H=32, KV=8, V=128256),
}
E2 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6.], device=dev)
TH = (E2[1:] + E2[:-1]) / 2
_H = torch.ones(1, 1, device=dev)
while _H.shape[0] < 16:
    _H = torch.cat([torch.cat([_H, _H], 1), torch.cat([_H, -_H], 1)], 0)
H16 = (_H / 4).to(torch.bfloat16)


def _nvfp4_quant(x, four_six=False, rot=False):
    M, K = x.shape
    if rot:
        x = (x.view(M, K // 16, 16) @ H16).view(M, K)
    xb = x.float().view(M, K // 16, 16)
    amax = xb.abs().amax(-1, keepdim=True).clamp(min=1e-12)

    def enc(s):
        t = (xb.abs() / s).clamp(max=6.0)
        idx = torch.bucketize(t, TH)
        return idx, (E2[idx] * s - xb.abs()).pow(2).sum(-1, keepdim=True)
    s = (amax / 6.0).to(torch.float8_e4m3fn).float()
    idx, err = enc(s)
    if four_six:
        s4 = (amax / 4.0).to(torch.float8_e4m3fn).float()
        idx4, err4 = enc(s4)
        pick = err4 < err
        idx = torch.where(pick, idx4, idx); s = torch.where(pick, s4, s)
    code = (idx | ((xb < 0).to(torch.int64) << 3)).to(torch.uint8).view(M, K)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous().view(torch.float4_e2m1fn_x2)
    return packed, s.to(torch.float8_e4m3fn).reshape(-1)


QF = {"nvfp4": torch.compile(lambda x: _nvfp4_quant(x), dynamic=False),
      "nvfp4_46": torch.compile(lambda x: _nvfp4_quant(x, True), dynamic=False),
      "nvfp4_rht": torch.compile(lambda x: _nvfp4_quant(x, False, True), dynamic=False)}


def _fp8_quant(x):
    s = x.abs().amax().float().clamp(min=1e-12) / 448.0
    return (x.float() / s).to(torch.float8_e4m3fn), s


fp8_quant = torch.compile(_fp8_quant, dynamic=False)
ONE = torch.ones((), device=dev)


class Lin:
    def __init__(self, K, N, mode):
        self.mode, self.K, self.N = mode, K, N
        if mode == "bf16":
            self.w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) / math.sqrt(K)
        elif mode == "fp8":
            self.w = (torch.randn(N, K, device=dev) * 0.5).to(torch.float8_e4m3fn); self.sw = torch.tensor(0.01, device=dev)
        else:
            self.w = torch.randint(0, 255, (N, K // 2), device=dev, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
            self.sw = torch.full((N * K // 16,), 0.01, device=dev).to(torch.float8_e4m3fn)
            self.qa = None

    def __call__(self, x):
        sh = x.shape; x = x.reshape(-1, sh[-1]); M = x.shape[0]
        if self.mode == "bf16":
            y = F.linear(x, self.w)
        elif self.mode == "fp8":
            a, s = fp8_quant(x)
            y = torch._scaled_mm(a, self.w.t(), scale_a=s, scale_b=self.sw, out_dtype=torch.bfloat16)
        else:
            Mp = max(128, (M + 127) // 128 * 128)
            if Mp != M:
                x = F.pad(x, (0, 0, 0, Mp - M))
            if self.mode == "nvfp4_noq":
                if self.qa is None or self.qa[0].shape[0] != Mp:   # pre-made FP4 activation (ideal fused producer)
                    self.qa = (torch.randint(0, 255, (Mp, self.K // 2), device=dev, dtype=torch.uint8).view(torch.float4_e2m1fn_x2),
                               torch.ones(Mp * self.K // 16, device=dev).to(torch.float8_e4m3fn))
                a, sa = self.qa
            else:
                a, sa = QF[self.mode](x)
            y = torch._scaled_mm(a, self.w.t(), scale_a=sa, scale_b=self.sw, out_dtype=torch.bfloat16)[:M]
        return y.view(*sh[:-1], self.N)


def rms(x, w):
    return F.rms_norm(x, (x.shape[-1],), w, 1e-6)


class Model:
    def __init__(self, c, mode, B, ctx):
        self.c = c; d, hd = c["d"], c["d"] // c["H"]; self.hd = hd
        self.emb = torch.randn(c["V"], d, device=dev, dtype=torch.bfloat16) * 0.02
        self.head = torch.randn(c["V"], d, device=dev, dtype=torch.bfloat16) * 0.02
        self.layers = []
        for _ in range(c["L"]):
            self.layers.append(dict(
                n1=torch.ones(d, device=dev, dtype=torch.bfloat16), n2=torch.ones(d, device=dev, dtype=torch.bfloat16),
                qkv=Lin(d, (c["H"] + 2 * c["KV"]) * hd, mode), o=Lin(c["H"] * hd, d, mode),
                gu=Lin(d, 2 * c["ffn"], mode), dn=Lin(c["ffn"], d, mode),
                kc=torch.randn(B, c["KV"], ctx, hd, device=dev, dtype=torch.bfloat16) if ctx else None,
                vc=torch.randn(B, c["KV"], ctx, hd, device=dev, dtype=torch.bfloat16) if ctx else None))
        self.fn = torch.ones(d, device=dev, dtype=torch.bfloat16)

    def forward(self, ids, decode):
        c, hd = self.c, self.hd
        x = F.embedding(ids, self.emb)
        B, T, _ = x.shape
        for l in self.layers:
            h = rms(x, l["n1"])
            qkv = l["qkv"](h)
            q, k, v = qkv.split([c["H"] * hd, c["KV"] * hd, c["KV"] * hd], -1)
            q = q.view(B, T, c["H"], hd).transpose(1, 2)
            k = k.view(B, T, c["KV"], hd).transpose(1, 2); v = v.view(B, T, c["KV"], hd).transpose(1, 2)
            if decode:   # write new kv at the last slot, attend over the full (fixed-length) cache
                l["kc"][:, :, -1:].copy_(k); l["vc"][:, :, -1:].copy_(v)
                a = F.scaled_dot_product_attention(q, l["kc"], l["vc"], enable_gqa=True)
            else:
                a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            x = x + l["o"](a.transpose(1, 2).reshape(B, T, c["H"] * hd))
            h = rms(x, l["n2"])
            g, u = l["gu"](h).chunk(2, -1)
            x = x + l["dn"](F.silu(g) * u)
        x = rms(x[:, -1:], self.fn)
        return F.linear(x, self.head)


def timed_graph(fn, n=20):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3): g.replay()
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    ts.sort(); return ts[len(ts) // 2]


if __name__ == "__main__":
    MODES = ["bf16", "fp8", "nvfp4_noq", "nvfp4", "nvfp4_46", "nvfp4_rht"]
    SCEN = [("prefill", 1, 2048, 0), ("prefill", 1, 8192, 0),
            ("decode", 1, 1, 4096), ("decode", 8, 1, 4096), ("decode", 32, 1, 4096), ("decode", 128, 1, 2048)]
    out = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "rows": []}
    print(out["device"], torch.__version__, flush=True)
    only = sys.argv[2:] if len(sys.argv) > 2 else list(CFGS)
    for name in only:
        c = CFGS[name]
        for kind, B, T, ctx in SCEN:
            row = dict(model=name, kind=kind, B=B, T=T, ctx=ctx)
            for mode in MODES:
                m = Model(c, mode, B, ctx)
                ids = torch.randint(0, c["V"], (B, T), device=dev)
                try:
                    with torch.no_grad():
                        row[mode] = round(timed_graph(lambda: m.forward(ids, kind == "decode")), 4)
                except Exception as e:
                    row[mode] = None; print("  ERR", mode, repr(e)[:200], flush=True)
                del m; torch.cuda.empty_cache()
            out["rows"].append(row)
            b = row["bf16"]
            print(f"{name:20s} {kind:7s} B={B:3d} T={T:5d} ctx={ctx:5d} | bf16 {b:.2f} ms | " +
                  "  ".join(f"{k} {row[k]:.2f} ({b / row[k]:.2f}x)" for k in MODES[1:] if row[k]), flush=True)
            json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "e2e_bench.json", "w"), indent=1)
