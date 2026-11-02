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
  sqgrid_46   SmoothQuant form s = ax^a / aw^(1-a), a searched per layer-group on {0,.1..1} by W4A4 output error (alpha oracle), 4/6
  awq_46      AWQ form s = mean|x|^a (normalised), a searched per layer-group on {0,.05..1} by W4A4 output error, 4/6
  let_46      OmniQuant-LET re-implementation: scale (init SmoothQuant a=0.5) + shift (init channel mid-range) on the attention and
              MLP inputs, scale on o_proj / down_proj inputs, trained per decoder block on the block-output MSE (BF16 inputs), 4/6;
              the shift is kept at inference (subtracted before quantization, W.delta added as a bias)
  letns_46    same without shift
--ste fmt: the STE target during scale learning is the evaluation quantizer itself (format scales, 4/6, rotation) instead of
fp32-scale E2M1. --seed n: calibration windows are taken n*(ncal+ngptq) windows further into the file.
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
D = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(D))
from fp4lib import E2M1, UNIF, BLK
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--hf", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--method", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--ncal", type=int, default=8); ap.add_argument("--Tcal", type=int, default=1024)
ap.add_argument("--steps", type=int, default=250); ap.add_argument("--limit", type=float, default=None)
ap.add_argument("--ngptq", type=int, default=128)   # GPTQ Hessian: ngptq x Tcal tokens (disjoint from the scale-learning set)
ap.add_argument("--fmt", default="nvfp4", choices=["nvfp4", "nvfp4_g32", "mxfp4"]); ap.add_argument("--gptq", action="store_true")
ap.add_argument("--ste", default="fp32", choices=["fp32", "fmt"]); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--calset", default="dclm", choices=["dclm", "wikitext"]); ap.add_argument("--omni_epochs", type=int, default=20); ap.add_argument("--logs_clamp", type=float, default=5.0); ap.add_argument("--dev_clamp", type=float, default=2.5)
ap.add_argument("--load_scales", default=""); ap.add_argument("--drop_shift", action="store_true"); ap.add_argument("--shift_skip_first", action="store_true"); ap.add_argument("--ppl_only", action="store_true")
ap.add_argument("--alpha", type=float, default=0.5); ap.add_argument("--save_scales", default=""); ap.add_argument("--let_steps", type=int, default=100)
ap.add_argument("--bs", type=int, default=16); ap.add_argument("--cal", default=os.environ.get("DCLM_VAL", "data/dclm_val.jsonl")); ap.add_argument("--tasks", default="hellaswag,arc_easy,arc_challenge,piqa,winogrande")
args = ap.parse_args()
dev = "cuda"; torch.manual_seed(args.seed)
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


def nvfp4(A, grid=E2M1, four_six=False, rot=False, orest=False):
    """A [N,K] -> dequantized fp32 in the original basis. 1xG blocks, absmax scales in the chosen format.
    four_six: also try amax/4.  orest: also try max(rest)/4 -- the block's largest element is treated as the outlier and the
    remaining values are spread over the fine 0..4 part of the grid (the outlier lands at <= 6 or is clipped). Per block the
    candidate with the lowest squared error is kept (E4M3 scales only for orest)."""
    N, K = A.shape
    Ab = A.float().view(N, K // G, G)
    if rot: Ab = Ab @ HG
    g = grid.to(A.device); th = (g[1:] + g[:-1]) / 2; gmax = g[-1]
    amax = Ab.abs().amax(-1, keepdim=True)
    gs = tensor_gs(amax, gmax, four_six or orest) if SCALE == "e4m3" else None
    cands = scale_cands(amax, gmax, four_six, gs)
    if orest and SCALE == "e4m3":
        rest = Ab.abs().topk(2, dim=-1).values[..., 1:2]
        c = ((rest / 4.0) / gs).clamp(max=448.0).to(torch.float8_e4m3fn).float() * gs
        cands.append(torch.where(c == 0, torch.ones_like(c), c))
    Ah = rnd(Ab, cands[0], g, th, gmax)
    if len(cands) > 1:
        e = (Ah - Ab).pow(2).sum(-1, keepdim=True)
        for c in cands[1:]:
            A2 = rnd(Ab, c, g, th, gmax); e2 = (A2 - Ab).pow(2).sum(-1, keepdim=True)
            Ah = torch.where(e2 < e, A2, Ah); e = torch.minimum(e, e2)
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


def ste(A):
    tgt = nvfp4(A.detach(), **QCFG[args.method]) if args.ste == "fmt" else e2m1_fp32(A.detach(), STE_ROT)
    return A + (tgt - A).detach()


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
        "sq_46": dict(four_six=True), "diag_46": dict(four_six=True), "diag": dict(), "diag_h": dict(rot=True), "diag_46_h": dict(four_six=True, rot=True),
        "sqgrid_46": dict(four_six=True), "awq_46": dict(four_six=True), "let_46": dict(four_six=True), "letns_46": dict(four_six=True),
        "e2m1_or": dict(orest=True), "e2m1_or46": dict(four_six=True, orest=True), "diag_or": dict(orest=True), "diag_or46": dict(four_six=True, orest=True), "awqneg_46": dict(four_six=True), "omni_46": dict(four_six=True), "omnins_46": dict(four_six=True),
        "diagomni_46": dict(four_six=True), "diagomni_or46": dict(four_six=True, orest=True), "sqomni_46": dict(four_six=True),
        "qdsomni_46": dict(four_six=True), "qdsblk_46": dict(four_six=True)}
SCALED = ("sq_46", "diag_46", "diag", "diag_h", "diag_46_h", "sqgrid_46", "awq_46", "let_46", "letns_46", "diag_or", "diag_or46", "awqneg_46")


class QLin(torch.nn.Module):
    def __init__(self, lin, qcfg, wq=None, shift=None):
        super().__init__()
        self.qcfg = qcfg
        self.register_buffer("b", None if lin.bias is None else lin.bias.detach().clone())
        W = lin.weight.data
        if wq is None: wq = torch.cat([nvfp4(W[i:i + 4096], **qcfg) for i in range(0, W.shape[0], 4096)])
        self.register_buffer("w", wq.to(dev, torch.bfloat16))
        self.shift = None
        if shift is not None:                      # LET shift: quantize (x - d), add W.d (full precision) as a bias
            self.shift = shift.to(dev, torch.bfloat16)
            b = W.float() @ shift.float().to(W.device); self.register_buffer("wd", b.to(dev, torch.bfloat16))
            self.b = (b if self.b is None else self.b.float() + b).to(torch.bfloat16)

    def forward(self, x):
        if self.shift is not None:
            if args.shift_skip_first and x.dim() == 3:          # causal test: leave position 0 (attention sink) unshifted
                sh = self.shift.expand_as(x).clone(); sh[:, 0] = 0; x = x - sh
                xs = x.reshape(-1, x.shape[-1])
                xq = torch.cat([nvfp4(xs[i:i + 8192], **self.qcfg) for i in range(0, xs.shape[0], 8192)]).to(torch.bfloat16)
                y = F.linear(xq, self.w, None).view(*x.shape[:-1], -1)
                y = y + (self.b.view(1, 1, -1) if self.b is not None else 0)
                y[:, 0] = y[:, 0] - self.wd.view(-1)                 # position 0 had no shift -> no compensation either
                return y
            x = x - self.shift
        xs = x.reshape(-1, x.shape[-1])
        xq = torch.cat([nvfp4(xs[i:i + 8192], **self.qcfg) for i in range(0, xs.shape[0], 8192)]).to(torch.bfloat16)
        y = F.linear(xq, self.w, self.b)
        return y.view(*x.shape[:-1], -1)


def linears(blk):
    a, m = blk.self_attn, blk.mlp
    return {"q_proj": a.q_proj, "k_proj": a.k_proj, "v_proj": a.v_proj, "o_proj": a.o_proj,
            "gate_proj": m.gate_proj, "up_proj": m.up_proj, "down_proj": m.down_proj}


GROUPS = [("attn_in", ("q_proj", "k_proj", "v_proj")), ("o", ("o_proj",)), ("mlp_in", ("gate_proj", "up_proj")), ("down", ("down_proj",))]


# ---------------------------------------------------------------- calibration data (DCLM val, same source as small-model runs)
_WT = None


def cal_batch(n=None, skip=0):
    n = args.ncal if n is None else n
    if args.calset == "wikitext":            # OmniQuant / GPTQ convention: random Tcal-token segments of WikiText-2 train
        global _WT
        if _WT is None:
            from datasets import load_dataset
            _WT = tok("\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"]), return_tensors="pt").input_ids[0]
        gen = torch.Generator().manual_seed(1000 * args.seed + skip)
        st = torch.randint(0, _WT.numel() - args.Tcal - 1, (n,), generator=gen)
        return torch.stack([_WT[a:a + args.Tcal] for a in st.tolist()]).to(dev)
    need = (skip + n) * args.Tcal; ids = []
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


def tie_o(s): return s.view(Hkv, REP, hd).mean(1, keepdim=True).expand(Hkv, REP, hd).reshape(-1)


def sq_scales(X, Ws, tie, alpha=None):
    a = args.alpha if alpha is None else alpha
    ax = X.abs().amax(0).clamp(min=1e-8); aw = torch.stack([W.abs().amax(0) for W in Ws]).amax(0).clamp(min=1e-8)
    s = ax.pow(a) / aw.pow(1 - a)
    return tie_o(s) if tie else s


@torch.no_grad()
def out_err(X, Ws, s):
    """W4A4 output error with the evaluation quantizer (format, 4/6, rotation) for scale vector s."""
    q = lambda A: torch.cat([nvfp4(A[i:i + 4096], **QCFG[args.method]) for i in range(0, A.shape[0], 4096)])
    Xq = q(X / s); e = 0.0
    for W in Ws:
        Y = X @ W.T; e += ((Xq @ q(W * s).T - Y).pow(2).sum() / Y.pow(2).sum()).item()
    return e


def grid_scales(X, Ws, tie, kind):
    """alpha search per layer-group: SmoothQuant form (sqgrid) or AWQ form (awq), selected by W4A4 output error."""
    if kind == "sqgrid": grid = [i / 10 for i in range(11)]; mk = lambda a: sq_scales(X, Ws, tie, a)
    else:
        # awq: a in [0,1] (spread X's outlier channels into W); awqneg also allows a < 0, i.e. s = mean|x|^a with a < 0
        # AMPLIFIES X's large channels (concentrates X's block energy onto its outlier channels) and shrinks them in W
        xm = X.abs().mean(0).clamp(min=1e-8)
        grid = [i / 20 for i in range(21)] if kind == "awq" else [i / 10 for i in range(-10, 11)]
        def mk(a):
            s = xm.pow(a).clamp(min=1e-4); s = s / (s.max() * s.min()).sqrt()
            return tie_o(s) if tie else s
    errs = [(out_err(X, Ws, mk(a)), a) for a in grid]
    e, a = min(errs)
    return mk(a), a, e


def capture_block(i, cal):
    """decoder block i: (input hidden, kwargs) per calibration sequence and the BF16 block output."""
    ins, outs = [], []
    h1 = LAYERS[i].register_forward_pre_hook(lambda m, a, k: ins.append(((a[0] if a else k["hidden_states"]).detach(),
                                                {kk: v for kk, v in k.items() if kk != "hidden_states"})), with_kwargs=True)
    h2 = LAYERS[i].register_forward_hook(lambda m, a, o: (outs.append((o[0] if isinstance(o, tuple) else o).detach()), (_ for _ in ()).throw(_Stop()))[1])
    with torch.no_grad():
        for s in cal:
            try: model(input_ids=s[None], use_cache=False)
            except _Stop: pass
    h1.remove(); h2.remove()
    return ins, outs


class TL(torch.nn.Module):
    """training-time LET wrapper: y = Q((x - d) / s) Q(W s)^T + W d (STE through the quantizer)."""
    def __init__(self, lin, sfn, dfn):
        super().__init__(); self.lin, self.sfn, self.dfn = lin, sfn, dfn

    def forward(self, x):
        s = self.sfn(); d = self.dfn() if self.dfn else None
        W = self.lin.weight.float(); xs = x.float()
        if d is not None: xs = xs - d
        sh = xs.shape; xs = xs.reshape(-1, sh[-1]) / s
        y = ste(xs) @ ste(W * s).T
        if d is not None: y = y + (W @ d)
        if self.lin.bias is not None: y = y + self.lin.bias.float()
        return y.view(*sh[:-1], -1).to(x.dtype)


def learn_let(i, cal, shift):
    """OmniQuant-style LET on decoder block i (block-output MSE, BF16 inputs). Returns (scales, shifts) per group."""
    blk = LAYERS[i]; L = linears(blk); ins, outs = capture_block(i, cal)
    X = capture(i, cal)
    P = {}
    for g, names in GROUPS:
        Ws = [L[n].weight.float() for n in names]
        v = torch.log(sq_scales(X[g], Ws, g == "o", 0.5))
        if g == "o": v = v.view(Hkv, REP, hd)[:, 0, :].reshape(-1)
        P[g] = [v.clone().requires_grad_(True), None]
        if shift and g in ("attn_in", "mlp_in"):
            d = (X[g].amax(0) + X[g].amin(0)) / 2          # OmniQuant init: channel mid-range
            P[g][1] = d.clone().requires_grad_(True)
    del X
    sfn = lambda g: (lambda: torch.exp(P[g][0].view(Hkv, 1, hd).expand(Hkv, REP, hd).reshape(-1) if g == "o" else P[g][0]))
    orig = {}
    for g, names in GROUPS:
        for n in names:
            parent = blk.self_attn if n in ("q_proj", "k_proj", "v_proj", "o_proj") else blk.mlp
            orig[n] = (parent, getattr(parent, n))
            setattr(parent, n, TL(getattr(parent, n), sfn(g), (lambda g=g: P[g][1]) if P[g][1] is not None else None))
    params = [p for g in P for p in P[g] if p is not None]
    opt = torch.optim.Adam(params, lr=5e-3); best = (1e30, None)
    for it in range(args.let_steps):
        tot = 0.0
        for (h, kw), tgt in zip(ins, outs):
            o = blk(h, **kw); o = o[0] if isinstance(o, tuple) else o
            loss = ((o.float() - tgt.float()) ** 2).mean() / (tgt.float() ** 2).mean(); loss.backward(); tot += loss.item()
        opt.step(); opt.zero_grad()
        if tot < best[0]: best = (tot, {g: (sfn(g)().detach().clone(), None if P[g][1] is None else P[g][1].detach().clone()) for g in P})
    for n, (parent, lin) in orig.items(): setattr(parent, n, lin)
    del ins, outs
    return {g: v[0] for g, v in best[1].items()}, {g: v[1] for g, v in best[1].items()}, best[0]


class TLW(torch.nn.Module):
    """OmniQuant training wrapper: y = Q((x - d)/s) Q(clip_c(W s))^T + W d; LWC clip_c = sigmoid(theta) * rowmax per output row."""
    def __init__(self, lin, sfn, dfn):
        super().__init__(); self.lin, self.sfn, self.dfn = lin, sfn, dfn
        self.theta = torch.full((lin.weight.shape[0], 1), 4.0, device=dev, requires_grad=True)

    def clipped(self, Ws):
        m = Ws.abs().amax(1, keepdim=True).detach() * torch.sigmoid(self.theta)
        return torch.maximum(torch.minimum(Ws, m), -m)

    def forward(self, x):
        s = self.sfn(); d = self.dfn() if self.dfn else None
        W = self.lin.weight.float(); xs = x.float()
        if d is not None: xs = xs - d
        sh = xs.shape; xs = xs.reshape(-1, sh[-1]) / s
        y = ste(xs) @ ste(self.clipped(W * s)).T
        if d is not None: y = y + (W @ d)
        if self.lin.bias is not None: y = y + self.lin.bias.float()
        return y.view(*sh[:-1], -1).to(x.dtype)


def omniquant(cal, shift):
    """OmniQuant (ICLR'24) on our NVFP4 quantizer: block-wise, sequential (quantized-model inputs vs FP targets), LET (scale
    [+ shift]) + LWC, AdamW (LET 1e-2, LWC 5e-3, wd 0), batch 1, args.omni_epochs epochs over the calibration segments.
    Blocks are replaced by QLin as they are finished; returns per-layer saved scales."""
    ins, kw = [], {}
    def pre(m, a, k):
        ins.append((a[0] if a else k["hidden_states"]).detach().to("cpu")); kw.update({kk: v for kk, v in k.items() if kk != "hidden_states"}); raise _Stop()
    h = LAYERS[0].register_forward_pre_hook(pre, with_kwargs=True)
    with torch.no_grad():
        for s in cal:
            try: model(input_ids=s[None], use_cache=False)
            except _Stop: pass
    h.remove()
    xfp = ins; xq = [x.clone() for x in ins]; SAVE = {}
    run = lambda blk, x: (lambda o: o[0] if isinstance(o, tuple) else o)(blk(x.to(dev), **kw))
    for i, blk in enumerate(LAYERS):
        t0b = time.time(); L = linears(blk)
        stats = {g: [None, None] for g, _ in GROUPS}; hs = []
        want = {"attn_in": L["q_proj"], "o": L["o_proj"], "mlp_in": L["gate_proj"], "down": L["down_proj"]}
        def st(g):
            def f(m, a):
                x = a[0].reshape(-1, a[0].shape[-1]).float(); mx, mn = x.amax(0), x.amin(0)
                stats[g][0] = mx if stats[g][0] is None else torch.maximum(stats[g][0], mx)
                stats[g][1] = mn if stats[g][1] is None else torch.minimum(stats[g][1], mn)
            return f
        for g, m in want.items(): hs.append(m.register_forward_pre_hook(st(g)))
        with torch.no_grad(): yfp = [run(blk, x).to("cpu") for x in xfp]
        for hh in hs: hh.remove()
        P = {}
        for g, names in GROUPS:
            ax = torch.maximum(stats[g][0].abs(), stats[g][1].abs()).clamp(min=1e-8)
            aw = torch.stack([L[n].weight.float().abs().amax(0) for n in names]).amax(0).clamp(min=1e-8)
            v = torch.log(ax.sqrt() / aw.sqrt())                                   # SmoothQuant init (a = 0.5)
            if g == "o": v = v.view(Hkv, REP, hd).mean(1).reshape(-1)
            d = ((stats[g][0] + stats[g][1]) / 2).clone().requires_grad_(True) if (shift and g in ("attn_in", "mlp_in")) else None   # OS+ init
            P[g] = [v.clone().requires_grad_(True), d]
        sfn = lambda g: (lambda: torch.exp(P[g][0].view(Hkv, 1, hd).expand(Hkv, REP, hd).reshape(-1) if g == "o" else P[g][0]))
        orig, W = {}, {}
        for g, names in GROUPS:
            for n in names:
                parent = blk.self_attn if n in ("q_proj", "k_proj", "v_proj", "o_proj") else blk.mlp
                orig[n] = (parent, getattr(parent, n)); W[n] = TLW(getattr(parent, n), sfn(g), (lambda g=g: P[g][1]) if P[g][1] is not None else None)
                setattr(parent, n, W[n])
        let_p = [p for g in P for p in P[g] if p is not None]; lwc_p = [w.theta for w in W.values()]
        V0 = {g: P[g][0].detach().clone() for g in P}
        opt = torch.optim.AdamW([{"params": let_p, "lr": 1e-2}, {"params": lwc_p, "lr": 5e-3}], weight_decay=0.0)
        gen = torch.Generator().manual_seed(args.seed + i); last = 0.0
        for ep in range(args.omni_epochs):
            tot = 0.0
            for j in torch.randperm(len(xq), generator=gen).tolist():
                o = run(blk, xq[j]); loss = F.mse_loss(o.float(), yfp[j].to(dev).float())
                opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
                with torch.no_grad():                   # stabilise log s: the block-absmax loss is invariant to a common factor
                    for g in P:                         # (random-walk direction) and blind to the NVFP4 tensor-scale floor, so
                        dv = P[g][0] - V0[g]; dv -= dv.mean()                    # per-sample updates drift; remove the common part
                        P[g][0].copy_(V0[g] + dv.clamp(-args.dev_clamp, args.dev_clamp))   # and bound the spread around the init
            last = tot / len(xq)
        with torch.no_grad():
            sc = {g: sfn(g)().detach().clone() for g in P}; sh = {g: (None if P[g][1] is None else P[g][1].detach().clone()) for g in P}
            clipc = {n: torch.sigmoid(w.theta).detach().clone() for n, w in W.items()}
        for n, (parent, lin) in orig.items(): setattr(parent, n, lin)
        fold(i, sc)
        with torch.no_grad():                     # bake LWC into the folded weights (same as clipping W*s in training)
            for n, lin in linears(blk).items():
                Wf = lin.weight.float(); m = Wf.abs().amax(1, keepdim=True) * clipc[n]
                lin.weight.copy_(torch.maximum(torch.minimum(Wf, m), -m).to(lin.weight.dtype))
        for g, names in GROUPS:
            for n in names:
                parent = blk.self_attn if n in ("q_proj", "k_proj", "v_proj", "o_proj") else blk.mlp
                shf = (sh[g] / sc[g]) if sh.get(g) is not None else None
                setattr(parent, n, QLin(getattr(parent, n), QCFG[M], None, shf))
        with torch.no_grad(): xq = [run(blk, x).to("cpu") for x in xq]
        xfp = yfp; torch.cuda.empty_cache()
        SAVE[i] = {g: sc[g].cpu() for g in sc}; SAVE[f"{i}.shift"] = {g: (v.cpu() if v is not None else None) for g, v in sh.items()}
        print(f"[{args.tag}/{M}] block {i}: final epoch loss {last:.3e} ({time.time()-t0b:.0f}s)", flush=True)
    return SAVE


def ours_seq(cal, mode):
    """Our methods on the OmniQuant data/budget: same calibration segments, args.omni_epochs epochs, batch 1 (one segment per
    step), AdamW lr 1e-2 wd 0 -- but OUR design: Linear-group output objective on FP inputs, diagonal scale only (init 1),
    folded, no shift, no LWC.  mode 'diag': learned;  mode 'sq': SmoothQuant a=0.5 from channel absmax over all segments."""
    ins, kw = [], {}
    def pre(m, a, k):
        ins.append((a[0] if a else k["hidden_states"]).detach().to("cpu")); kw.update({kk: v for kk, v in k.items() if kk != "hidden_states"}); raise _Stop()
    h = LAYERS[0].register_forward_pre_hook(pre, with_kwargs=True)
    with torch.no_grad():
        for sg in cal:
            try: model(input_ids=sg[None], use_cache=False)
            except _Stop: pass
    h.remove()
    xfp = ins; SAVE = {}
    run = lambda blk, x: (lambda o: o[0] if isinstance(o, tuple) else o)(blk(x.to(dev), **kw))
    for i, blk in enumerate(LAYERS):
        t0b = time.time(); L = linears(blk)
        want = {"attn_in": L["q_proj"], "o": L["o_proj"], "mlp_in": L["gate_proj"], "down": L["down_proj"]}
        got = {}
        hs = [m.register_forward_pre_hook(lambda m, a, g=g: got.__setitem__(g, a[0].reshape(-1, a[0].shape[-1]).float())) for g, m in want.items()]
        Ws = {g: [L[n].weight.float() for n in names] for g, names in GROUPS}
        if args.load_scales and mode == "diag":                     # e.g. diagomni_or46: reuse diagomni_46's scales, new quantizer
            sc = {g: v.to(dev) for g, v in torch.load(args.load_scales, map_location="cpu")["scales"][i].items()}; last = 0.0
        elif mode in ("qds", "qdsblk"):
            amax = {}
            with torch.no_grad():
                for x in xfp:
                    run(blk, x)
                    for g, X in got.items(): amax[g] = X.abs().amax(0) if g not in amax else torch.maximum(amax[g], X.abs().amax(0))
            aw = {g: torch.stack([W.abs().amax(0) for W in Ws[g]]).amax(0).clamp(min=1e-8) for g, _ in GROUPS}
            def qds_s(g, a):
                v = amax[g].clamp(min=1e-8).pow(a) / aw[g].pow(1 - a); return tie_o(v) if g == "o" else v
            grid = [k / 10 for k in range(11)]; sel = list(range(min(16, len(xfp))))
            fitq = lambda A: e2m1_fp32(A)                              # QDS fitting quantizer: E2M1, FP32 block scale, absmax -> 6
            if mode == "qds":                                           # per-group GEMM output error (the QDS definition)
                err = {g: [0.0] * len(grid) for g, _ in GROUPS}; den = {g: 0.0 for g, _ in GROUPS}
                with torch.no_grad():
                    for j in sel:
                        run(blk, xfp[j])
                        for g, _ in GROUPS:
                            X = got[g]; Ys = [X @ W.T for W in Ws[g]]; den[g] += sum((Y * Y).sum().item() for Y in Ys)
                            for k, a in enumerate(grid):
                                s_ = qds_s(g, a); Xq = fitq(X / s_)
                                err[g][k] += sum(((Xq @ fitq(W * s_).T - Y) ** 2).sum().item() for W, Y in zip(Ws[g], Ys))
                alpha = {g: grid[min(range(len(grid)), key=lambda k: err[g][k])] for g, _ in GROUPS}
            else:                                                       # block-output criterion, coordinate descent over groups
                class QS(torch.nn.Module):
                    def __init__(self, lin): super().__init__(); self.lin = lin; self.s = None
                    def forward(self, x):
                        sh = x.shape; xs = x.reshape(-1, sh[-1]).float() / self.s
                        y = fitq(xs) @ fitq(self.lin.weight.float() * self.s).T
                        if self.lin.bias is not None: y = y + self.lin.bias.float()
                        return y.view(*sh[:-1], -1).to(x.dtype)
                orig = {}
                for g, names in GROUPS:
                    for n in names:
                        parent = blk.self_attn if n in ("q_proj", "k_proj", "v_proj", "o_proj") else blk.mlp
                        orig[n] = (parent, getattr(parent, n), g); setattr(parent, n, QS(getattr(parent, n)))
                with torch.no_grad():
                    for hh in hs: hh.remove()
                    hs = []
                    for n, (parent, lin, g) in orig.items(): pass
                    tgt = {j: None for j in sel}
                    for n, (parent, lin, g) in orig.items(): setattr(parent, n, lin)
                    for j in sel: tgt[j] = run(blk, xfp[j]).float()
                    for n, (parent, lin, g) in orig.items(): setattr(parent, n, QS(lin))
                    alpha = {g: 0.5 for g, _ in GROUPS}
                    def set_s():
                        for n, (parent, lin, g) in orig.items(): getattr(parent, n).s = qds_s(g, alpha[g])
                    def berr():
                        set_s(); return sum(((run(blk, xfp[j]).float() - tgt[j]) ** 2).sum().item() for j in sel)
                    for _pass in range(2):
                        for g, _ in GROUPS:
                            best = (1e30, alpha[g])
                            for a in grid:
                                alpha[g] = a; e = berr()
                                if e < best[0]: best = (e, a)
                            alpha[g] = best[1]
                for n, (parent, lin, g) in orig.items(): setattr(parent, n, lin)
            sc = {g: qds_s(g, alpha[g]) for g, _ in GROUPS}; last = 0.0
            ALPHA_SEQ[i] = alpha
        elif mode == "sq":
            amax = {}
            with torch.no_grad():
                for x in xfp:
                    run(blk, x)
                    for g, X in got.items(): amax[g] = X.abs().amax(0) if g not in amax else torch.maximum(amax[g], X.abs().amax(0))
            sc = {}
            for g, _ in GROUPS:
                aw = torch.stack([W.abs().amax(0) for W in Ws[g]]).amax(0).clamp(min=1e-8)
                v = amax[g].clamp(min=1e-8).sqrt() / aw.sqrt(); sc[g] = tie_o(v) if g == "o" else v
            last = 0.0
        else:
            V = {g: torch.zeros(Hkv * hd if g == "o" else Ws[g][0].shape[1], device=dev, requires_grad=True) for g, _ in GROUPS}
            sfn = lambda g: torch.exp(V[g].view(Hkv, 1, hd).expand(Hkv, REP, hd).reshape(-1) if g == "o" else V[g])
            opt = torch.optim.AdamW(list(V.values()), lr=1e-2, weight_decay=0.0)
            gen = torch.Generator().manual_seed(args.seed + i); last = 0.0
            for ep in range(args.omni_epochs):
                tot = 0.0
                for j in torch.randperm(len(xfp), generator=gen).tolist():
                    with torch.no_grad(): run(blk, xfp[j])                       # captures the four group inputs
                    loss = 0.0
                    for g, _ in GROUPS:
                        X = got[g]; s_ = sfn(g); Xq = ste(X / s_)
                        for W in Ws[g]:
                            Y = X @ W.T; loss = loss + ((Xq @ ste(W * s_).T - Y) ** 2).sum() / (Y * Y).sum()
                    opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
                    with torch.no_grad():                # same stabilisation as omniquant(): zero-mean, bounded log s
                        for v_ in V.values(): v_.sub_(v_.mean()).clamp_(-args.dev_clamp, args.dev_clamp)
                last = tot / len(xfp)
            sc = {g: sfn(g).detach().clone() for g, _ in GROUPS}
        for hh in hs: hh.remove()
        with torch.no_grad(): yfp = [run(blk, x).to("cpu") for x in xfp]       # FP inputs for the next block (our design)
        fold(i, sc); xfp = yfp; SAVE[i] = {g: sc[g].cpu() for g in sc}; torch.cuda.empty_cache()
        print(f"[{args.tag}/{M}] block {i}: final epoch loss {last:.4f} ({time.time()-t0b:.0f}s)", flush=True)
    return SAVE


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
OMNI = M in ("omni_46", "omnins_46")
OURSEQ = M in ("diagomni_46", "diagomni_or46", "sqomni_46", "qdsomni_46", "qdsblk_46")
ALPHA_SEQ = {}
if OMNI or OURSEQ:
    _sv = omniquant(cal_batch(), M == "omni_46") if OMNI else ours_seq(cal_batch(), {"sqomni_46": "sq", "qdsomni_46": "qds", "qdsblk_46": "qdsblk"}.get(M, "diag"))
    res["cal_sec"] = time.time() - t0; res["alpha"] = {str(k): v for k, v in ALPHA_SEQ.items()}
    if args.save_scales: torch.save(dict(scales=_sv, args=vars(args)), args.save_scales)
elif M in SCALED or (args.gptq and M != "bf16"):
    off = args.seed * (args.ncal + args.ngptq)
    cal = cal_batch(skip=off); fit = {}; ALPHA = {}; SHIFT = {}; SAVE = {}
    if args.gptq: cal_g = cal_batch(args.ngptq, skip=off + args.ncal)   # 8k tokens leave the K=14336 Hessian rank-deficient -> GPTQ overfits
    for i in range(len(LAYERS)):
        L = linears(LAYERS[i])
        if M in ("let_46", "letns_46"):
            if args.load_scales:
                LS = torch.load(args.load_scales, map_location="cpu")["scales"]
                sc = {g: v.to(dev) for g, v in LS[i].items()}; sh = {g: (None if v is None else v.to(dev)) for g, v in LS.get(f"{i}.shift", {}).items()}
                if args.drop_shift: sh = {}                 # keep LET's (shift-co-trained) scales, discard the shift
            else:
                sc, sh, fit[f"{i}"] = learn_let(i, cal, M == "let_46")
            fold(i, sc)
            for g, names in GROUPS:                    # shift lives in the folded basis: d' = d / s
                if sh.get(g) is not None:
                    for n in names: SHIFT[(i, n)] = (sh[g] / sc[g]).detach()
            SAVE[i] = {g: sc[g].cpu() for g in sc}; SAVE[f"{i}.shift"] = {g: (v.cpu() if v is not None else None) for g, v in sh.items()}
        elif M in SCALED:
            X = capture(i, cal); sc = {}
            for g, names in GROUPS:
                Ws = [L[n].weight.float() for n in names]
                if M == "sq_46": sc[g] = sq_scales(X[g], Ws, g == "o")
                elif M in ("sqgrid_46", "awq_46", "awqneg_46"): sc[g], ALPHA[f"{i}.{g}"], fit[f"{i}.{g}"] = grid_scales(X[g], Ws, g == "o", {"awq_46": "awq", "awqneg_46": "awqneg"}.get(M, "sqgrid"))
                else: sc[g], fit[f"{i}.{g}"] = learn_scales(X[g], Ws, g == "o")
            del X; fold(i, sc); SAVE[i] = {g: sc[g].cpu() for g in sc}
        if args.gptq:
            HH = capture_hessian(i, cal_g)              # inputs after this layer's fold
            for g, names in GROUPS:
                Hs = HH.pop(g)
                for n in names: WQ[(i, n)] = gptq(L[n].weight, Hs, **QCFG[M]).to(torch.bfloat16).cpu()   # park on CPU: 8B x2 does not fit 24 GB
                del Hs; torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        print(f"[{args.tag}/{args.fmt}/{M}{'+gptq' if args.gptq else ''}] layer {i} done ({time.time()-t0:.0f}s)", flush=True)
    res["cal_fit"] = fit; res["cal_sec"] = time.time() - t0; res["alpha"] = ALPHA
    if args.save_scales: torch.save(dict(scales=SAVE, alpha=ALPHA, args=vars(args)), args.save_scales)
if M != "bf16" and not OMNI:
    for i, blk in enumerate(LAYERS):
        for n, lin in linears(blk).items():
            parent = blk.self_attn if n in ("q_proj", "k_proj", "v_proj", "o_proj") else blk.mlp
            setattr(parent, n, QLin(lin, QCFG[M], WQ.pop((i, n), None), SHIFT.get((i, n)) if M == "let_46" else None))
    torch.cuda.empty_cache()
res["wikitext2_ppl"] = wikitext_ppl(); print(f"[{args.tag}/{args.fmt}/{M}{'+gptq' if args.gptq else ''}] wikitext2 ppl {res['wikitext2_ppl']:.3f}", flush=True)
json.dump(res, open(args.out, "w"), indent=1)
if args.ppl_only: sys.exit(0)
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
