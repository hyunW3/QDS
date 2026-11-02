"""W4A4 NVFP4 PTQ of 7-8B open models: does foldable learned channel scaling + 4/6 beat rotation (H16, UFP4)?

One method per process. All decoder Linears are W4A4 (FPROP); embedding / lm_head stay BF16.
Quantizer: 1x16 blocks, E4M3 block scales with an FP32 per-tensor scale (NVFP4), RTN; weights quantized once, activations
quantized on the fly (per token rows); the dequantized operands are multiplied in BF16 (exact for these values).
Methods
  bf16        reference
  e2m1        plain NVFP4
  e2m1_46     NVFP4 with per-block 4/6 scale choice (Four Over Six)
  e2m1_h16    16x16 block Hadamard on both operands along K, then NVFP4 (rotation baseline)
  ufp4        uniform E1M2 grid (= symmetric INT4) + H16 (arXiv 2606.20381)
  sq_46       SmoothQuant alpha=0.5, folded into RMSNorm / v_proj / up_proj, then 4/6
  diag_46     ours: learned per-channel scale (shared per input, GQA-tied for o_proj), layer-wise output-MSE objective with STE,
              folded exactly like SmoothQuant (zero inference overhead), then 4/6
  diag        same objective, plain NVFP4 (no 4/6)
  diag_46_h   diag_h scales evaluated with 4/6 + H
  diag_h      same, but scales learned through (and evaluated with) the GxG block Hadamard quantizer: diag + H, no 4/6
Formats (--fmt): nvfp4 = G16 + E4M3 scales (+FP32 tensor scale); nvfp4_g32 = G32 + E4M3 (non-native, block-size control);
mxfp4 = G32 + E8M0 power-of-two scales (OCP: 2^(floor(log2 amax) - floor(log2 gmax)), values above gmax clipped);
for E8M0 the "4/6" choice is between that exponent and exponent+1 (no clipping), picked per block by MSE.
Rotations (h16 / ufp4) use a GxG block Hadamard. --gptq: weights rounded with GPTQ (same block format, per-row group
scales fixed when the group is reached, damp 0.01, no act-order) on the transformed BF16 model's inputs (non-sequential);
for rotated methods GPTQ runs in the rotated basis.
Eval: WikiText-2 test perplexity (seq 2048) and lm-eval 0-shot HellaSwag / ARC-E / ARC-C / PIQA / WinoGrande.
"""
import argparse, json, math, os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # GPTQ + scale learning fragment 24 GB cards      # models / datasets are pre-cached on every node (gated Llama tokens differ per node)
from pathlib import Path
import torch, torch.nn.functional as F
D = Path(__file__).resolve().parent; sys.path.insert(0, str(D))
from fp4lib import E2M1, UNIF, BLK
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--hf", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--method", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--ncal", type=int, default=8); ap.add_argument("--Tcal", type=int, default=1024)
ap.add_argument("--steps", type=int, default=250); ap.add_argument("--limit", type=float, default=None)
ap.add_argument("--ngptq", type=int, default=128)   # GPTQ Hessian: ngptq x Tcal tokens (disjoint from the scale-learning set)
ap.add_argument("--fmt", default="nvfp4", choices=["nvfp4", "nvfp4_g32", "mxfp4"]); ap.add_argument("--gptq", action="store_true")
ap.add_argument("--bs", type=int, default=16); ap.add_argument("--cal", default=os.environ.get("DCLM_VAL", "data/dclm_val.jsonl")); ap.add_argument("--tasks", default="hellaswag,arc_easy,arc_challenge,piqa,winogrande")
args = ap.parse_args()
dev = "cuda"; torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(args.hf)
model = AutoModelForCausalLM.from_pretrained(args.hf, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
for p in model.parameters(): p.requires_grad_(False)
cfg = model.config
Hq, Hkv = cfg.num_attention_heads, cfg.num_key_value_heads
hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // Hq
REP = Hq // Hkv
LAYERS = model.model.layers
G = 16 if args.fmt == "nvfp4" else 32
SCALE = "e8m0" if args.fmt == "mxfp4" else "e4m3"
_H = torch.ones(1, 1, device=dev)
while _H.shape[0] < G: _H = torch.cat([torch.cat([_H, _H], 1), torch.cat([_H, -_H], 1)], 0)
HG = _H / math.sqrt(G)


# ---------------------------------------------------------------- quantizers
def tensor_gs(amax, gmax, four_six):          # FP32 tensor scale so that E4M3 block scales fit (max 448)
    return (amax.max() / (4.0 if four_six else gmax) / 448.0).clamp(min=1e-30)


def scale_cands(amax, gmax, four_six, gs):
    if SCALE == "e4m3":
        out = [((amax / d) / gs).clamp(max=448.0).to(torch.float8_e4m3fn).float() * gs for d in ([gmax, 4.0] if four_six else [gmax])]   # saturate (GPTQ updates can exceed the tensor-scale range; E4M3 has no inf -> NaN)
    else:
        e = torch.floor(torch.log2(amax.clamp(min=1e-30))) - math.floor(math.log2(float(gmax)))
        out = [torch.exp2(e + k) for k in ([0, 1] if four_six else [0])]
    return [torch.where(c == 0, torch.ones_like(c), c) for c in out]


def rnd(A, s, g, th, gmax):
    return torch.sign(A) * g[torch.bucketize((A.abs() / s).clamp(max=gmax), th)] * s


def nvfp4(A, grid=E2M1, four_six=False, rot=False):
    """A [N,K] -> dequantized fp32 in the original basis. 1xG blocks, absmax scales in the chosen format."""
    N, K = A.shape
    Ab = A.float().view(N, K // G, G)
    if rot: Ab = Ab @ HG
    g = grid.to(A.device); th = (g[1:] + g[:-1]) / 2; gmax = g[-1]
    amax = Ab.abs().amax(-1, keepdim=True)
    cands = scale_cands(amax, gmax, four_six, tensor_gs(amax, gmax, four_six) if SCALE == "e4m3" else None)
    Ah = rnd(Ab, cands[0], g, th, gmax)
    if four_six:
        A4 = rnd(Ab, cands[1], g, th, gmax)
        e6 = (Ah - Ab).pow(2).sum(-1, keepdim=True); e4 = (A4 - Ab).pow(2).sum(-1, keepdim=True)
        Ah = torch.where(e4 < e6, A4, Ah)
    if rot: Ah = Ah @ HG.T
    return Ah.view(N, K)


def e2m1_fp32(A, rot=False):   # calibration STE target: E2M1, fp32 block scales, block G (G=16 = the validated small-model runs)
    N, K = A.shape; Ab = A.float().view(N, K // G, G); g = E2M1.to(A.device); th = (g[1:] + g[:-1]) / 2
    if rot: Ab = Ab @ HG
    s = Ab.abs().amax(-1, keepdim=True) / 6.0; s = torch.where(s == 0, torch.ones_like(s), s)
    Ah = torch.sign(Ab) * g[torch.bucketize((Ab.abs() / s).clamp(max=6), th)] * s
    if rot: Ah = Ah @ HG.T
    return Ah.view(N, K)


STE_ROT = args.method in ("diag_h", "diag_46_h")          # diag_h learns its scales through the rotated quantizer


def ste(A): return A + (e2m1_fp32(A.detach(), STE_ROT) - A).detach()


@torch.no_grad()
def gptq(W, Hs, grid=E2M1, four_six=False, rot=False, damp=0.01, B=128):
    """GPTQ with 1xG block-format groups. W [N,K] (original basis), Hs = X^T X [K,K]. Returns dequantized W (original basis)."""
    W = W.float().clone(); N, K = W.shape; Hs = Hs.clone()
    if rot:
        W = (W.view(N, K // G, G) @ HG).view(N, K)
        Hs = (Hs.view(K, K // G, G) @ HG).view(K, K)              # R^T Hs R with R = blockdiag(HG), low-memory
        Hs = (HG.T @ Hs.view(K // G, G, K)).view(K, K)
    g = grid.to(W.device); th = (g[1:] + g[:-1]) / 2; gmax = g[-1]
    gs = tensor_gs(W.view(N, K // G, G).abs().amax(-1), gmax, four_six) if SCALE == "e4m3" else None
    dead = torch.diag(Hs) == 0; Hs[dead, dead] = 1; W[:, dead] = 0
    Hs += damp * torch.diag(Hs).mean() * torch.eye(K, device=W.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hs)), upper=True)
    Q = torch.zeros_like(W)
    for i1 in range(0, K, B):
        i2 = min(i1 + B, K); W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); E1 = torch.zeros_like(W1); Hi = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            if i % G == 0:                     # fix this group's per-row scale from the current (error-updated) weights
                grp = W1[:, i:i + G]; amax = grp.abs().amax(-1, keepdim=True)
                cands = scale_cands(amax, gmax, four_six, gs); s = cands[0]
                if four_six:
                    e0 = (rnd(grp, cands[0], g, th, gmax) - grp).pow(2).sum(-1, keepdim=True)
                    e1 = (rnd(grp, cands[1], g, th, gmax) - grp).pow(2).sum(-1, keepdim=True)
                    s = torch.where(e1 < e0, cands[1], cands[0])
            w = W1[:, i]; q = rnd(w[:, None], s, g, th, gmax)[:, 0]
            Q1[:, i] = q; err = (w - q) / Hi[i, i]
            W1[:, i:] -= err[:, None] @ Hi[i, i:][None]; E1[:, i] = err
        Q[:, i1:i2] = Q1; W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    if rot: Q = (Q.view(N, K // G, G) @ HG.T).view(N, K)
    return Q


M = args.method
QCFG = {"e2m1": dict(), "e2m1_46": dict(four_six=True), "e2m1_h16": dict(rot=True), "e2m1_h": dict(rot=True), "ufp4": dict(grid=UNIF, rot=True),
        "sq_46": dict(four_six=True), "diag_46": dict(four_six=True), "diag": dict(), "diag_h": dict(rot=True), "diag_46_h": dict(four_six=True, rot=True)}


class QLin(torch.nn.Module):
    def __init__(self, lin, qcfg, wq=None):
        super().__init__()
        self.qcfg = qcfg; self.bias = lin.bias
        W = lin.weight.data
        if wq is None: wq = torch.cat([nvfp4(W[i:i + 4096], **qcfg) for i in range(0, W.shape[0], 4096)])
        self.register_buffer("w", wq.to(dev, torch.bfloat16))

    def forward(self, x):
        xs = x.reshape(-1, x.shape[-1])
        xq = torch.cat([nvfp4(xs[i:i + 8192], **self.qcfg) for i in range(0, xs.shape[0], 8192)]).to(torch.bfloat16)
        y = F.linear(xq, self.w, self.bias)
        return y.view(*x.shape[:-1], -1)


def linears(blk):
    a, m = blk.self_attn, blk.mlp
    return {"q_proj": a.q_proj, "k_proj": a.k_proj, "v_proj": a.v_proj, "o_proj": a.o_proj,
            "gate_proj": m.gate_proj, "up_proj": m.up_proj, "down_proj": m.down_proj}


GROUPS = [("attn_in", ("q_proj", "k_proj", "v_proj")), ("o", ("o_proj",)), ("mlp_in", ("gate_proj", "up_proj")), ("down", ("down_proj",))]


# ---------------------------------------------------------------- calibration data (DCLM val, same source as small-model runs)
def cal_batch(n=None, skip=0):
    n = args.ncal if n is None else n; need = (skip + n) * args.Tcal; ids = []
    for line in open(args.cal):
        ids += [tok.bos_token_id] * (tok.bos_token_id is not None) + tok.encode(json.loads(line)["text"], add_special_tokens=False)
        if len(ids) >= need: break
    assert len(ids) >= need, (len(ids), need)
    return torch.tensor(ids[skip * args.Tcal:need], device=dev).view(n, args.Tcal)


class _Stop(Exception): pass


@torch.no_grad()
def capture(i, cal):
    """inputs of layer i's four Linear groups (BF16 model; earlier folds are function-preserving)."""
    L = linears(LAYERS[i]); want = {"attn_in": L["q_proj"], "o": L["o_proj"], "mlp_in": L["gate_proj"], "down": L["down_proj"]}
    got = {g: [] for g in want}; hs = []
    for g, mod in want.items():
        hs.append(mod.register_forward_pre_hook(lambda m, a, g=g: got[g].append(a[0].reshape(-1, a[0].shape[-1]).float())))
    hs.append(LAYERS[i].register_forward_hook(lambda *a: (_ for _ in ()).throw(_Stop())))
    for s in cal:
        try: model(input_ids=s[None], use_cache=False)
        except _Stop: pass
    for h in hs: h.remove()
    return {g: torch.cat(v) for g, v in got.items()}


@torch.no_grad()
def capture_hessian(i, cal):
    """sum_x x^T x of layer i's four Linear-group inputs, accumulated without storing activations."""
    L = linears(LAYERS[i]); want = {"attn_in": L["q_proj"], "o": L["o_proj"], "mlp_in": L["gate_proj"], "down": L["down_proj"]}
    Hs = {g: None for g in want}; hs = []
    def acc(g):
        def f(m, a):
            x = a[0].reshape(-1, a[0].shape[-1]).float(); h = x.T @ x
            Hs[g] = h if Hs[g] is None else Hs[g].add_(h)
        return f
    for g, mod in want.items(): hs.append(mod.register_forward_pre_hook(acc(g)))
    hs.append(LAYERS[i].register_forward_hook(lambda *a: (_ for _ in ()).throw(_Stop())))
    for s in cal:
        try: model(input_ids=s[None], use_cache=False)
        except _Stop: pass
    for h in hs: h.remove()
    return Hs


def learn_scales(X, Ws, tie):
    K = X.shape[1]
    v = torch.zeros(Hkv * hd if tie else K, device=dev, requires_grad=True)
    Ys = [X @ W.T for W in Ws]; nYs = [(Y * Y).sum() for Y in Ys]
    opt = torch.optim.Adam([v], lr=1e-2); best = (1e30, None)
    expand = lambda u: u.view(Hkv, 1, hd).expand(Hkv, REP, hd).reshape(-1) if tie else u
    for _ in range(args.steps):
        s = torch.exp(expand(v)); Xq = ste(X / s)
        loss = sum(((Xq @ ste(W * s).T - Y) ** 2).sum() / nY for W, Y, nY in zip(Ws, Ys, nYs))
        opt.zero_grad(); loss.backward(); opt.step()
        if loss.item() < best[0]: best = (loss.item(), torch.exp(expand(v.detach())))
    return best[1], best[0]


def sq_scales(X, Ws, tie):
    ax = X.abs().amax(0).clamp(min=1e-8); aw = torch.stack([W.abs().amax(0) for W in Ws]).amax(0).clamp(min=1e-8)
    s = (ax.sqrt() / aw.sqrt())
    if tie: s = s.view(Hkv, REP, hd).mean(1, keepdim=True).expand(Hkv, REP, hd).reshape(-1)
    return s


@torch.no_grad()
def fold(i, sc):
    blk = LAYERS[i]; L = linears(blk)
    s_in, s_o, s_mlp, s_dn = sc["attn_in"], sc["o"], sc["mlp_in"], sc["down"]
    blk.input_layernorm.weight.div_(s_in.to(torch.bfloat16))
    for n in ("q_proj", "k_proj", "v_proj"): L[n].weight.mul_(s_in.to(torch.bfloat16))
    blk.post_attention_layernorm.weight.div_(s_mlp.to(torch.bfloat16))
    for n in ("gate_proj", "up_proj"): L[n].weight.mul_(s_mlp.to(torch.bfloat16))
    s_kv = s_o.view(Hkv, REP, hd)[:, 0, :].reshape(-1)
    L["v_proj"].weight.div_(s_kv[:, None].to(torch.bfloat16))
    if L["v_proj"].bias is not None: L["v_proj"].bias.div_(s_kv.to(torch.bfloat16))
    L["o_proj"].weight.mul_(s_o.to(torch.bfloat16))
    L["up_proj"].weight.div_(s_dn[:, None].to(torch.bfloat16))
    L["down_proj"].weight.mul_(s_dn.to(torch.bfloat16))


# ---------------------------------------------------------------- eval
@torch.no_grad()
def wikitext_ppl(T=2048):
    from datasets import load_dataset
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    n = ids.numel() // T; nll = 0.0
    for j in range(n):
        s = ids[j * T:(j + 1) * T].to(dev)[None]
        nll += F.cross_entropy(model(input_ids=s, use_cache=False).logits[0, :-1].float(), s[0, 1:], reduction="sum").item()
    return math.exp(nll / (n * (T - 1)))


res = dict(model=args.tag, method=M, fmt=args.fmt, gptq=args.gptq, hf=args.hf, args=vars(args))
t0 = time.time(); WQ = {}
if M in ("sq_46", "diag_46", "diag", "diag_h", "diag_46_h") or (args.gptq and M != "bf16"):
    cal = cal_batch(); fit = {}
    if args.gptq: cal_g = cal_batch(args.ngptq, skip=args.ncal)   # 8k tokens leave the K=14336 Hessian rank-deficient -> GPTQ overfits
    for i in range(len(LAYERS)):
        L = linears(LAYERS[i])
        if M in ("sq_46", "diag_46", "diag", "diag_h", "diag_46_h"):
            X = capture(i, cal); sc = {}
            for g, names in GROUPS:
                Ws = [L[n].weight.float() for n in names]
                if M == "sq_46": sc[g] = sq_scales(X[g], Ws, g == "o")
                else: sc[g], fit[f"{i}.{g}"] = learn_scales(X[g], Ws, g == "o")
            del X; fold(i, sc)
        if args.gptq:
            HH = capture_hessian(i, cal_g)              # inputs after this layer's fold
            for g, names in GROUPS:
                Hs = HH.pop(g)
                for n in names: WQ[(i, n)] = gptq(L[n].weight, Hs, **QCFG[M]).to(torch.bfloat16).cpu()   # park on CPU: 8B x2 does not fit 24 GB
                del Hs; torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        print(f"[{args.tag}/{args.fmt}/{M}{'+gptq' if args.gptq else ''}] layer {i} done ({time.time()-t0:.0f}s)", flush=True)
    res["cal_fit"] = fit; res["cal_sec"] = time.time() - t0
if M != "bf16":
    for i, blk in enumerate(LAYERS):
        for n, lin in linears(blk).items():
            parent = blk.self_attn if n in ("q_proj", "k_proj", "v_proj", "o_proj") else blk.mlp
            setattr(parent, n, QLin(lin, QCFG[M], WQ.pop((i, n), None)))
    torch.cuda.empty_cache()
res["wikitext2_ppl"] = wikitext_ppl(); print(f"[{args.tag}/{args.fmt}/{M}{'+gptq' if args.gptq else ''}] wikitext2 ppl {res['wikitext2_ppl']:.3f}", flush=True)
json.dump(res, open(args.out, "w"), indent=1)
import lm_eval
from lm_eval.models.huggingface import HFLM
lm = HFLM(pretrained=model, tokenizer=tok, batch_size=args.bs)
r = lm_eval.simple_evaluate(model=lm, tasks=args.tasks.split(","), num_fewshot=0, limit=args.limit, bootstrap_iters=0)
acc = {}
for t, v in r["results"].items():
    acc[t] = {k.split(",")[0]: v[k] for k in v if k.startswith("acc")}
res["lm_eval"] = acc
main = {t: (a.get("acc_norm", a.get("acc")) if t != "winogrande" else a["acc"]) for t, a in acc.items()}
res["avg5"] = sum(main.values()) / len(main)
print(f"[{args.tag}/{args.fmt}/{M}{'+gptq' if args.gptq else ''}] " + "  ".join(f"{t} {100*v:.2f}" for t, v in main.items()) + f"  | avg {100*res['avg5']:.2f}  ({time.time()-t0:.0f}s)", flush=True)
json.dump(res, open(args.out, "w"), indent=1)
