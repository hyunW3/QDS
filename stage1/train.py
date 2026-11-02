"""Stage 1: small-LM pretraining with emulated FP4 GEMMs (fake quant), one arm per run.

Every nn.Linear inside the decoder layers runs its three training GEMMs through FP4 emulation:
  FPROP  Y  = X W^T      operands X [N,K], W [O,K]        blocks along K
  DGRAD  dX = dY W       operands dY [N,O], W^T [K,O]     blocks along O
  WGRAD  dW = dY^T X     operands dY^T [O,N], X^T [K,N]   blocks along N (tokens)
1x16 blocks along the reduction axis, absmax scale (FP32), dequantize, matmul in BF16. Embedding, lm_head, attention
BMMs and norms stay BF16. Any transform on the reduction axis is applied to both operands (the product is unchanged).
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D.parent))
from fp4lib import E2M1, UNIF, BLK, hadamard

ARMS = {
    # name: per-GEMM dict(grid, scale, tf, dy) ; tf in none|h16|rperm ; dy rounding rtn|sr ; ofp8 = outlier FP8 fraction
    "bf16": None,
    "nvfp4": dict(grid="e2m1", scale="6", tf={"fprop": "none", "dgrad": "none", "wgrad": "h16"}, dy="sr", ofp8=0.0),
    "e2m1_h16all": dict(grid="e2m1", scale="6", tf="h16", dy="sr", ofp8=0.0),
    "ufp4": dict(grid="unif", scale="6", tf="h16", dy="sr", ofp8=0.0),
    "e2m1_46": dict(grid="e2m1", scale="46", tf="none", dy="rtn", ofp8=0.0),
    "e2m1_46_rperm": dict(grid="e2m1", scale="46", tf="rperm", dy="rtn", ofp8=0.0),
    "e2m1_ofp8": dict(grid="e2m1", scale="6", tf="none", dy="rtn", ofp8=0.01),
    "e2m1_rtn": dict(grid="e2m1", scale="6", tf="none", dy="rtn", ofp8=0.0),
    # one GEMM type in FP8 (both operands, E4M3 1x16), the other two in FP4 (E2M1 RTN, no transform)
    "e2m1_fprop8": dict(grid={"fprop": "fp8", "dgrad": "e2m1", "wgrad": "e2m1"}, scale="6", tf="none", dy="rtn", ofp8=0.0),
    "e2m1_dgrad8": dict(grid={"fprop": "e2m1", "dgrad": "fp8", "wgrad": "e2m1"}, scale="6", tf="none", dy="rtn", ofp8=0.0),
    "e2m1_wgrad8": dict(grid={"fprop": "e2m1", "dgrad": "e2m1", "wgrad": "fp8"}, scale="6", tf="none", dy="rtn", ofp8=0.0),
    # dynamic SmoothQuant-style diagonal scaling on every GEMM's reduction axis (computed per call, no calibration)
    "e2m1_smooth": dict(grid="e2m1", scale="6", tf="smooth", dy="rtn", ofp8=0.0),
    "e2m1_46_smooth": dict(grid="e2m1", scale="46", tf="smooth", dy="rtn", ofp8=0.0),
    # same as e2m1_rtn / e2m1_46 / e2m1_smooth but with stochastic rounding on dY (as in nvfp4 / ufp4) -> separates
    # the effect of SR from the effect of RHT. NB: 4/6 picks the lower-MSE scale per block, which biases SR.
    "e2m1_sr": dict(grid="e2m1", scale="6", tf="none", dy="sr", ofp8=0.0),
    "e2m1_46_sr": dict(grid="e2m1", scale="46", tf="none", dy="sr", ofp8=0.0),
    "e2m1_smooth_sr": dict(grid="e2m1", scale="6", tf="smooth", dy="sr", ofp8=0.0),
    "e2m1_46_smooth_sr": dict(grid="e2m1", scale="46", tf="smooth", dy="sr", ofp8=0.0),
    # deployable versions of e2m1_46_smooth_sr ("smooth" is an oracle: it reads the column absmax of the very tensors it
    # then quantizes). smooth_delay: s from this call site's previous-step stats; smooth_learn: s learned online per site.
    "e2m1_46_smoothdelay_sr": dict(grid="e2m1", scale="46", tf="smooth_delay", dy="sr", ofp8=0.0),
    "e2m1_46_smoothlearn_sr": dict(grid="e2m1", scale="46", tf="smooth_learn", dy="sr", ofp8=0.0),
    # e2m1_46_smoothdelay_sr + the NVIDIA recipe's RHT on the WGRAD inputs only: does WGRAD RHT add anything on top?
    "e2m1_46_smoothdelay_sr_wrht": dict(grid="e2m1", scale="46", tf={"fprop": "smooth_delay", "dgrad": "smooth_delay", "wgrad": "h16"}, dy="sr", ofp8=0.0),
}

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=list(ARMS))
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=8000)
ap.add_argument("--micro", type=int, default=16)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--seqlen", type=int, default=1024)
ap.add_argument("--lr", type=float, default=1.5e-3)
ap.add_argument("--warmup", type=int, default=400)
ap.add_argument("--eval_every", type=int, default=500)
ap.add_argument("--dim", type=int, default=512); ap.add_argument("--layers", type=int, default=12)
ap.add_argument("--ffn", type=int, default=1536); ap.add_argument("--heads", type=int, default=8)
ap.add_argument("--slearn_lr", type=float, default=1e-2)   # smooth_learn: Adam lr on the log-scale v
ap.add_argument("--slearn_rows", type=int, default=1024)  # smooth_learn: row subsample per operand for the objective
args = ap.parse_args()
dev = "cuda"
torch.manual_seed(args.seed); np.random.seed(args.seed)
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
CFG = ARMS[args.arm]
H = hadamard(BLK, dev)
qgen = torch.Generator(device=dev); qgen.manual_seed(1234 + args.seed)

# ---------------------------------------------------------------- quantizers (reduction axis = last dim)


def _levels(t, grid, rnd):
    """RTN / SR onto the magnitude grid by arithmetic (identical to nearest-level search, much faster).
    E2M1 steps: 0.5 on [0,2), 1 on [2,4), 2 on [4,6].  Uniform (E1M2): 0.25 on [0,1.75]."""
    if grid == "e2m1":
        d = torch.where(t < 2, 0.5, torch.where(t < 4, 1.0, 2.0))
        top = 6.0
    else:
        d = torch.full_like(t, 0.25)
        top = 1.75
    if rnd == "sr":
        lo = torch.floor(t / d) * d
        u = torch.rand(t.shape, device=dev, generator=qgen)
        q = lo + d * (u < (t - lo) / d).float()
    else:
        q = torch.round(t / d) * d
    return q.clamp(max=top)


def fp4(A, grid, scale, rnd):
    N, R = A.shape
    Ab = A.float().view(N, R // BLK, BLK)
    gmax = 6.0 if grid == "e2m1" else 1.75
    amax = Ab.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    sgn, mag = torch.sign(Ab), Ab.abs()
    s6 = amax / gmax
    q6 = sgn * _levels((mag / s6).clamp(max=gmax), grid, rnd) * s6
    if scale == "46" and grid == "e2m1":
        s4 = amax / 4.0
        q4 = sgn * _levels((mag / s4).clamp(max=6.0), grid, rnd) * s4
        e6 = (q6 - Ab).pow(2).sum(-1, keepdim=True); e4 = (q4 - Ab).pow(2).sum(-1, keepdim=True)
        q6 = torch.where(e4 < e6, q4, q6)
    return q6.view(N, R)


def fp8(A):
    N, R = A.shape
    Ab = A.float().view(N, R // BLK, BLK)
    s = (Ab.abs().amax(-1, keepdim=True) / 448.0).clamp(min=1e-30)
    return ((Ab / s).to(torch.float8_e4m3fn).float() * s).view(N, R)


# fused elementwise quantizers (20-50x faster than eager; differ from eager only on ~0.05% exact-boundary ties)
import torch._dynamo
torch._dynamo.config.cache_size_limit = 256            # many (shape, grid, scale, rounding) specializations
torch._dynamo.config.accumulated_cache_size_limit = 4096
fp4 = torch.compile(fp4, dynamic=False)
# fp8 stays eager: Triton on Ampere (sm86) has no fp8e4nv


# ---------------------------------------------------------------- deployable smoothing (smooth_delay / smooth_learn)
# Both keep per-call-site state; site = (decoder layer . Linear name, GEMM role), i.e. 12 x 7 x 3 = 252 independent sites.
# Eval forwards (SM["train"] False) use the stored s and never update stats / v.
#   smooth_delay  s = sqrt(sa/sb) (same formula as the oracle) from the column absmax this site saw in the PREVIOUS
#                 optimizer step (elementwise max over that step's grad-accum micro-batches = the previous step's full
#                 batch). Step 0 falls back to the current call's stats.
#   smooth_learn  s = exp(v), v [R] per site, initialised from the SmoothQuant formula on the site's first call. Each call
#                 uses the current v (no peeking), then takes ONE Adam step (own optimizer) on
#                 L = ||Q(A/s) Q(B s)^T - A B^T||^2 / ||A B^T||^2 on detached operands, <= slearn_rows random rows of each
#                 operand, same grid / scale / rounding as the arm. Q = STE through the rounding only (per-block absmax
#                 stays differentiable, so the quantization error scales with the operand as it should).
#                 Diagnostic every 100 steps: L of the oracle formula s (current tensors) on the same row subsample.
SM = dict(train=True, diag=False)
SM_STATS = {}        # site -> dict(prev=(sa, sb) | None, acc=(sa, sb) | None)
SM_LEARN = {}        # site -> (v, Adam state)
SM_LOSS = []         # smooth_learn objective of each training call in the current step (logged as the step mean)
SM_LOSS0 = []        # diagnostic: same objective with the oracle s
lgen = torch.Generator(device=dev); lgen.manual_seed(4321 + args.seed)


def _colmax(A, B):
    return A.float().abs().amax(0).clamp(min=1e-12), B.float().abs().amax(0).clamp(min=1e-12)


def smooth_delay_scale(A, B, key):
    st = SM_STATS.setdefault(key, dict(prev=None, acc=None))
    if not SM["train"]:
        sa, sb = st["prev"] if st["prev"] is not None else _colmax(A, B)
        return (sa / sb).sqrt()
    cur = _colmax(A, B)
    st["acc"] = cur if st["acc"] is None else (torch.maximum(st["acc"][0], cur[0]), torch.maximum(st["acc"][1], cur[1]))
    sa, sb = st["prev"] if st["prev"] is not None else cur
    return (sa / sb).sqrt()


def smooth_roll():
    """End of an optimizer step: this step's accumulated stats become the next step's s."""
    for st in SM_STATS.values():
        st["prev"], st["acc"] = st["acc"], None


def _ste_q(x, grid, scale, rnd):
    N, R = x.shape
    with torch.no_grad():
        q = fp4(x.detach().contiguous(), grid, scale, rnd)
    xb = x.view(N, R // BLK, BLK)
    am = xb.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    return (xb + am * ((q.view_as(xb) - xb.detach()) / am.detach())).view(N, R)


def _rows(n):
    k = args.slearn_rows
    return None if n <= k else torch.randperm(n, device=dev, generator=lgen)[:k]


def _slearn_obj(As, Bs, v, grid, scale, ra):
    ref = (As.bfloat16() @ Bs.bfloat16().T).float()
    s = v.exp()
    P = (_ste_q(As / s, grid, scale, ra).bfloat16() @ _ste_q(Bs * s, grid, scale, "rtn").bfloat16().T).float()
    return (P - ref).pow(2).sum() / ref.pow(2).sum().clamp(min=1e-30)


_slearn_obj = torch.compile(_slearn_obj, dynamic=False)   # fuses fwd+bwd elementwise work (~2x faster than eager)


def _adam(v, g, st, b1=0.9, b2=0.999, eps=1e-8):
    """Plain bias-corrected Adam on one vector (no weight decay); a torch.optim.Adam per site is CPU-bound here."""
    st["t"] += 1; m, u = st["m"], st["u"]
    m.mul_(b1).add_(g, alpha=1 - b1); u.mul_(b2).addcmul_(g, g, value=1 - b2)
    v.data.addcdiv_(m, (u / (1 - b2 ** st["t"])).sqrt_().add_(eps), value=-args.slearn_lr / (1 - b1 ** st["t"]))


def smooth_learn_scale(A, B, key, grid, scale, ra):
    if key not in SM_LEARN:
        sa, sb = _colmax(A, B)
        v = (0.5 * (sa.log() - sb.log())).clone().requires_grad_(True)
        SM_LEARN[key] = (v, dict(t=0, m=torch.zeros_like(v), u=torch.zeros_like(v)))
    v, st = SM_LEARN[key]
    sc = v.detach().exp()                       # the s used by this call: chosen before looking at the update
    if SM["train"]:
        ia, ib = _rows(A.shape[0]), _rows(B.shape[0])
        As = (A if ia is None else A[ia]).detach().float(); Bs = (B if ib is None else B[ib]).detach().float()
        if SM["diag"]:
            with torch.no_grad():
                sa, sb = _colmax(A, B)
                SM_LOSS0.append(_slearn_obj(As, Bs, 0.5 * (sa.log() - sb.log()), grid, scale, ra))
        with torch.enable_grad():
            L = _slearn_obj(As, Bs, v, grid, scale, ra)
            g, = torch.autograd.grad(L, v)
        _adam(v, g, st)
        SM_LOSS.append(L.detach())
    return sc


def qgemm(A, B, gemm, a_is_dy, site=None):
    """A [M,R], B [P,R] -> A_hat @ B_hat^T with this arm's recipe (R = reduction axis). site: call-site id (stateful tf)."""
    c = CFG
    tf = c["tf"][gemm] if isinstance(c["tf"], dict) else c["tf"]
    grid = c["grid"][gemm] if isinstance(c["grid"], dict) else c["grid"]
    if grid == "fp8":
        return fp8(A.contiguous()).bfloat16() @ fp8(B.contiguous()).bfloat16().T
    R = A.shape[1]
    if tf == "h16":
        sg = (torch.randint(0, 2, (R,), device=dev, generator=qgen) * 2 - 1).float()
        A = ((A.float() * sg).view(A.shape[0], -1, BLK) @ H).view(A.shape)
        B = ((B.float() * sg).view(B.shape[0], -1, BLK) @ H).view(B.shape)
    elif tf == "smooth":
        sa = A.float().abs().amax(0).clamp(min=1e-12); sb = B.float().abs().amax(0).clamp(min=1e-12)
        sc = (sa / sb).sqrt()
        A, B = A.float() / sc, B.float() * sc
    elif tf in ("smooth_delay", "smooth_learn"):
        assert site is not None
        key = (site, gemm)
        if tf == "smooth_delay":
            sc = smooth_delay_scale(A, B, key)
        else:
            sc = smooth_learn_scale(A, B, key, grid, c["scale"], c["dy"] if a_is_dy else "rtn")
        A, B = A.float() / sc, B.float() * sc
    elif tf == "rperm":
        p = torch.randperm(R, device=dev, generator=qgen)
        A, B = A[:, p], B[:, p]
    ra = c["dy"] if a_is_dy else "rtn"
    if c["ofp8"] > 0:
        n = max(BLK, int(math.ceil(c["ofp8"] * R / BLK)) * BLK)
        score = A.float().pow(2).mean(0) * B.float().pow(2).mean(0)
        idx = score.topk(n).indices
        mask = torch.ones(R, dtype=torch.bool, device=dev); mask[idx] = False
        rest = mask.nonzero().squeeze(1)
        Y = fp8(A[:, idx].contiguous()).bfloat16() @ fp8(B[:, idx].contiguous()).bfloat16().T
        Y = Y + fp4(A[:, rest].contiguous(), grid, c["scale"], ra).bfloat16() @ fp4(B[:, rest].contiguous(), grid, c["scale"], "rtn").bfloat16().T
        return Y
    return fp4(A.contiguous(), grid, c["scale"], ra).bfloat16() @ fp4(B.contiguous(), grid, c["scale"], "rtn").bfloat16().T


class FP4LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, site):
        ctx.save_for_backward(x, w); ctx.site = site
        xs = x.reshape(-1, x.shape[-1])
        y = qgemm(xs, w, "fprop", False, site)
        return y.view(*x.shape[:-1], w.shape[0]).to(x.dtype)

    @staticmethod
    def backward(ctx, gy):
        x, w = ctx.saved_tensors
        xs = x.reshape(-1, x.shape[-1]); dy = gy.reshape(-1, gy.shape[-1])
        dx = qgemm(dy, w.t(), "dgrad", True, ctx.site).view(*x.shape).to(x.dtype)
        dw = qgemm(dy.t(), xs.t(), "wgrad", True, ctx.site).to(w.dtype)
        return dx, dw, None


class FP4Linear(torch.nn.Module):
    def __init__(self, lin, site):
        super().__init__()
        assert lin.bias is None
        self.weight = lin.weight; self.site = site          # site: "layer.linear", e.g. "3.q_proj"

    def forward(self, x):
        return FP4LinearFn.apply(x.to(torch.bfloat16), self.weight.to(torch.bfloat16), self.site)


# ---------------------------------------------------------------- model
cfg = LlamaConfig(vocab_size=49152, hidden_size=args.dim, intermediate_size=args.ffn, num_hidden_layers=args.layers,
                  num_attention_heads=args.heads, num_key_value_heads=args.heads, max_position_embeddings=args.seqlen,
                  rope_theta=10000.0, tie_word_embeddings=True, attention_bias=False, mlp_bias=False)
model = LlamaForCausalLM(cfg)
if CFG is not None:
    for li, layer in enumerate(model.model.layers):
        for parent in (layer.self_attn, layer.mlp):
            for name, child in list(parent.named_children()):
                if isinstance(child, torch.nn.Linear):
                    setattr(parent, name, FP4Linear(child, f"{li}.{name}"))
model = model.to(dev)
nparam = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1, fused=True)

# ---------------------------------------------------------------- data
train = np.memmap(D / "data/train.bin", dtype=np.uint16, mode="r")
val = np.memmap(D / "data/val.bin", dtype=np.uint16, mode="r")
rng = np.random.default_rng(args.seed)
VAL_SEQ = 256
val_x = torch.from_numpy(np.stack([val[i * (args.seqlen + 1):(i + 1) * (args.seqlen + 1)].astype(np.int64) for i in range(VAL_SEQ)]))


def batch():
    off = rng.integers(0, len(train) - args.seqlen - 1, size=args.micro)
    x = np.stack([train[o:o + args.seqlen + 1].astype(np.int64) for o in off])
    return torch.from_numpy(x).to(dev, non_blocking=True)


def lm_loss(xb):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=xb[:, :-1]).logits
    return F.cross_entropy(logits.float().flatten(0, 1), xb[:, 1:].flatten())


@torch.no_grad()
def evaluate():
    model.eval(); SM["train"] = False; tot = 0.0
    for i in range(0, VAL_SEQ, args.micro):
        tot += lm_loss(val_x[i:i + args.micro].to(dev)).item() * min(args.micro, VAL_SEQ - i)
    model.train(); SM["train"] = True; return tot / VAL_SEQ


def lr_at(s):
    if s < args.warmup:
        return args.lr * (s + 1) / args.warmup
    p = (s - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


log = open(out / "log.jsonl", "a")
meta = dict(vars(args), cfg=CFG, params=nparam, tokens_per_step=args.micro * args.accum * args.seqlen)
(out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
print(f"[{args.arm}] params {nparam/1e6:.1f}M, tokens/step {meta['tokens_per_step']}", flush=True)
t0 = time.time(); model.train()
for step in range(args.steps):
    for gpar in opt.param_groups:
        gpar["lr"] = lr_at(step)
    tl = 0.0; SM["diag"] = (step + 1) % 100 == 0
    for _ in range(args.accum):
        loss = lm_loss(batch()) / args.accum
        loss.backward(); tl += loss.item()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
    opt.step(); opt.zero_grad(set_to_none=True)
    smooth_roll()
    rec = dict(step=step + 1, loss=tl, gnorm=gn, lr=lr_at(step), t=time.time() - t0)
    if SM_LOSS:
        rec["slearn_L"] = torch.stack(SM_LOSS).mean().item(); SM_LOSS.clear()
    if SM_LOSS0:
        rec["slearn_L_oracle"] = torch.stack(SM_LOSS0).mean().item(); SM_LOSS0.clear()
    if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
        rec["val"] = evaluate()
        print(f"[{args.arm}] step {step+1} loss {tl:.4f} val {rec['val']:.4f} gnorm {gn:.2f} {(time.time()-t0)/(step+1):.2f}s/step", flush=True)
    if not math.isfinite(tl):
        rec["diverged"] = True; log.write(json.dumps(rec) + "\n"); break
    log.write(json.dumps(rec) + "\n"); log.flush()
torch.save(model.state_dict(), out / "final.pt")
print(f"[{args.arm}] done in {(time.time()-t0)/3600:.2f} h" + (f", {len(SM_STATS) or len(SM_LEARN)} smoothing sites" if SM_STATS or SM_LEARN else ""), flush=True)
