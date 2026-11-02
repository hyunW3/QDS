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
  diag        same scales, plain NVFP4 (ablation)
Eval: WikiText-2 test perplexity (seq 2048) and lm-eval 0-shot HellaSwag / ARC-E / ARC-C / PIQA / WinoGrande.
"""
import os
import argparse, json, math, sys, time
from pathlib import Path
import torch, torch.nn.functional as F
D = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(D))
from fp4lib import E2M1, UNIF, BLK
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--hf", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--method", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--ncal", type=int, default=8); ap.add_argument("--Tcal", type=int, default=1024)
ap.add_argument("--steps", type=int, default=250); ap.add_argument("--limit", type=float, default=None)
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
_H = torch.ones(1, 1, device=dev)
while _H.shape[0] < BLK: _H = torch.cat([torch.cat([_H, _H], 1), torch.cat([_H, -_H], 1)], 0)
H16 = _H / math.sqrt(BLK)


# ---------------------------------------------------------------- quantizers
def nvfp4(A, grid=E2M1, four_six=False, rot=False):
    """A [N,K] -> dequantized fp32. Block absmax 1x16, E4M3 block scales with FP32 tensor scale."""
    N, K = A.shape
    Ab = A.float().view(N, K // BLK, BLK)
    if rot: Ab = Ab @ H16
    g = grid.to(A.device); th = (g[1:] + g[:-1]) / 2; gmax = g[-1]
    amax = Ab.abs().amax(-1, keepdim=True)
    gs = (amax.max() / (4.0 if four_six else gmax) / 448.0).clamp(min=1e-30)   # tensor scale so block scales fit E4M3

    def enc(div):
        s = ((amax / div) / gs).to(torch.float8_e4m3fn).float() * gs
        s = torch.where(s == 0, torch.ones_like(s), s)
        Ah = torch.sign(Ab) * g[torch.bucketize((Ab.abs() / s).clamp(max=gmax), th)] * s
        return Ah
    Ah = enc(gmax)
    if four_six:
        A4 = enc(4.0)
        e6 = (Ah - Ab).pow(2).sum(-1, keepdim=True); e4 = (A4 - Ab).pow(2).sum(-1, keepdim=True)
        Ah = torch.where(e4 < e6, A4, Ah)
    if rot: Ah = Ah @ H16.T
    return Ah.view(N, K)


def e2m1_fp32(A):              # calibration STE target (same as the validated small-model runs)
    N, K = A.shape; Ab = A.float().view(N, K // BLK, BLK); g = E2M1.to(A.device); th = (g[1:] + g[:-1]) / 2
    s = Ab.abs().amax(-1, keepdim=True) / 6.0; s = torch.where(s == 0, torch.ones_like(s), s)
    return (torch.sign(Ab) * g[torch.bucketize((Ab.abs() / s).clamp(max=6), th)] * s).view(N, K)


def ste(A): return A + (e2m1_fp32(A.detach()) - A).detach()


M = args.method
QCFG = {"e2m1": dict(), "e2m1_46": dict(four_six=True), "e2m1_h16": dict(rot=True), "ufp4": dict(grid=UNIF, rot=True),
        "sq_46": dict(four_six=True), "diag_46": dict(four_six=True), "diag": dict()}


class QLin(torch.nn.Module):
    def __init__(self, lin, qcfg):
        super().__init__()
        self.qcfg = qcfg; self.bias = lin.bias
        W = lin.weight.data
        self.register_buffer("w", torch.cat([nvfp4(W[i:i + 4096], **qcfg) for i in range(0, W.shape[0], 4096)]).to(torch.bfloat16))

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
def cal_batch():
    ids = []
    for line in open(args.cal):
        ids += [tok.bos_token_id] * (tok.bos_token_id is not None) + tok.encode(json.loads(line)["text"], add_special_tokens=False)
        if len(ids) >= args.ncal * args.Tcal: break
    return torch.tensor(ids[:args.ncal * args.Tcal], device=dev).view(args.ncal, args.Tcal)


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


res = dict(model=args.tag, method=M, hf=args.hf, args=vars(args))
t0 = time.time()
if M in ("sq_46", "diag_46", "diag"):
    cal = cal_batch(); fit = {}
    for i in range(len(LAYERS)):
        X = capture(i, cal); L = linears(LAYERS[i]); sc = {}
        for g, names in GROUPS:
            Ws = [L[n].weight.float() for n in names]
            if M == "sq_46": sc[g] = sq_scales(X[g], Ws, g == "o")
            else: sc[g], fit[f"{i}.{g}"] = learn_scales(X[g], Ws, g == "o")
        del X; fold(i, sc)
        print(f"[{args.tag}/{M}] layer {i} calibrated ({time.time()-t0:.0f}s)", flush=True)
    res["cal_fit"] = fit; res["cal_sec"] = time.time() - t0
if M != "bf16":
    for blk in LAYERS:
        for n, lin in linears(blk).items():
            parent = blk.self_attn if n in ("q_proj", "k_proj", "v_proj", "o_proj") else blk.mlp
            setattr(parent, n, QLin(lin, QCFG[M]))
    torch.cuda.empty_cache()
res["wikitext2_ppl"] = wikitext_ppl(); print(f"[{args.tag}/{M}] wikitext2 ppl {res['wikitext2_ppl']:.3f}", flush=True)
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
print(f"[{args.tag}/{M}] " + "  ".join(f"{t} {100*v:.2f}" for t, v in main.items()) + f"  | avg {100*res['avg5']:.2f}  ({time.time()-t0:.0f}s)", flush=True)
json.dump(res, open(args.out, "w"), indent=1)
