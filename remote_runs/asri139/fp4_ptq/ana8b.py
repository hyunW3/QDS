"""Why channel scaling instead of rotation for block-scaled E2M1: per-GEMM error anatomy and loss sensitivity on an 8B model.

For every decoder layer and Linear group (attn_in = q/k/v input, o, mlp_in = gate/up input, down) of a BF16 model, the group
input X (eval windows, disjoint from calibration) and the member weights W are transformed by
  none | sq (SmoothQuant a=0.5) | learned (diagonal s from a ptq8b.py --save_scales run) | rot (GxG block Hadamard) | learned+rot
and quantized with   nvfp4_46 (G16, E4M3 + 4/6) | nvfp4 (G16, E4M3) | mxfp4 (G32, E8M0, OCP floor rule).
Per (layer, group, transform, quantizer) we record
  - tensor NMSE of X and of each W (in the transformed basis; rotations are orthonormal),
  - GEMM output NMSE ||Q(X')Q(W')^T - X W^T||^2 / ||X W^T||^2,
  - squared-error share by normalised-magnitude bin t = |a| / s_block: dead (t < 0.25, rounds to 0), fine [0.25, 2),
    mid [2, 4) (spacing 1), top [4, gmax] (spacing 2), clip (t > gmax, MXFP4 floor rule),
  - block crest factor amax / rms and scale utilisation amax / (s * gmax) (E8M0 wastes range),
and, for the scale vectors, how the learned s relates to the SmoothQuant s and whether it equalises channel magnitudes inside
16-channel blocks.
Sensitivity: the NLL increase on the eval windows when ONLY that group is quantized (W4A4, nvfp4_46), per transform.
"""
import argparse, json, math, os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--hf", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--scales", required=True, help="ptq8b.py --save_scales file with the learned diagonal (nvfp4 diag_46)")
ap.add_argument("--cal", default="dclm_val_head2000.jsonl"); ap.add_argument("--nwin", type=int, default=4); ap.add_argument("--T", type=int, default=1024)
ap.add_argument("--skip", type=int, default=600, help="eval windows start this many windows into the file (calibration uses the first ~400)")
ap.add_argument("--layers", default="all"); ap.add_argument("--no_sens", action="store_true")
args = ap.parse_args()
dev = "cuda"; torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(args.hf)
model = AutoModelForCausalLM.from_pretrained(args.hf, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
for p in model.parameters(): p.requires_grad_(False)
cfg = model.config; Hq, Hkv = cfg.num_attention_heads, cfg.num_key_value_heads
hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // Hq; REP = Hq // Hkv
LAYERS = model.model.layers
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6.], device=dev)
GROUPS = [("attn_in", ("q_proj", "k_proj", "v_proj")), ("o", ("o_proj",)), ("mlp_in", ("gate_proj", "up_proj")), ("down", ("down_proj",))]
QUANTS = {"nvfp4_46": (16, "e4m3", True), "nvfp4": (16, "e4m3", False), "mxfp4": (32, "e8m0", False)}
HAD = {}


def hadamard(G):
    if G not in HAD:
        H = torch.ones(1, 1, device=dev)
        while H.shape[0] < G: H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        HAD[G] = H / math.sqrt(G)
    return HAD[G]


def quant(A, G, sfmt, four_six):
    """A [N,K] fp32 -> (A_hat, t = |a|/s, s per block [N,K/G,1]) for E2M1 with 1xG absmax blocks."""
    N, K = A.shape; Ab = A.view(N, K // G, G); g = E2M1; th = (g[1:] + g[:-1]) / 2; gmax = 6.0
    amax = Ab.abs().amax(-1, keepdim=True)
    if sfmt == "e4m3":
        gs = (amax.max() / (4.0 if four_six else gmax) / 448.0).clamp(min=1e-30)
        cands = [((amax / d) / gs).clamp(max=448.0).to(torch.float8_e4m3fn).float() * gs for d in ([gmax, 4.0] if four_six else [gmax])]
    else:
        e = torch.floor(torch.log2(amax.clamp(min=1e-30))) - 2.0
        cands = [torch.exp2(e + k) for k in ([0, 1] if four_six else [0])]
    cands = [torch.where(c == 0, torch.ones_like(c), c) for c in cands]
    def rnd(s): return torch.sign(Ab) * g[torch.bucketize((Ab.abs() / s).clamp(max=gmax), th)] * s
    s = cands[0]; Ah = rnd(s)
    if four_six:
        A4 = rnd(cands[1]); pick = (A4 - Ab).pow(2).sum(-1, keepdim=True) < (Ah - Ab).pow(2).sum(-1, keepdim=True)
        Ah = torch.where(pick, A4, Ah); s = torch.where(pick, cands[1], s)
    return Ah.view(N, K), (Ab.abs() / s).view(N, K), s


BINS = [("dead", 0, 0.25), ("fine", 0.25, 2), ("mid", 2, 4), ("top", 4, 6.0001), ("clip", 6.0001, 1e30)]


def anatomy(A, G, sfmt, four_six):
    Ah, t, s = quant(A, G, sfmt, four_six)
    e2 = (Ah - A).pow(2); tot = e2.sum()
    out = dict(nmse=(tot / A.pow(2).sum()).item())
    for name, lo, hi in BINS:
        m = (t >= lo) & (t < hi)
        out[f"share_{name}"] = (e2[m].sum() / tot).item() if tot > 0 else 0.0
        out[f"mass_{name}"] = m.float().mean().item()
    N, K = A.shape; Ab = A.view(N, K // G, G); amax = Ab.abs().amax(-1); rms = Ab.pow(2).mean(-1).sqrt().clamp(min=1e-30)
    crest = (amax / rms).flatten(); util = (amax / (s.squeeze(-1) * 6.0)).flatten()
    out.update(crest_mean=crest.mean().item(), crest_p95=crest.quantile(0.95).item() if crest.numel() < 16_000_000 else crest[::7].quantile(0.95).item(),
               util_mean=util.mean().item())
    return out, Ah


def block_hetero(X, W, s, G=16):
    """channel-magnitude spread inside 16-channel blocks: mean over blocks of std(log absmax_channel) for X and W."""
    def spread(cm):  # cm [K] channel absmax
        b = cm.clamp(min=1e-12).log().view(-1, G); return b.std(-1).mean().item()
    return dict(x_spread=spread((X / s).abs().amax(0)), w_spread=spread((W * s).abs().amax(0)))


def cal_windows():
    need = (args.skip + args.nwin) * args.T; ids = []
    for line in open(args.cal):
        ids += [tok.bos_token_id] * (tok.bos_token_id is not None) + tok.encode(json.loads(line)["text"], add_special_tokens=False)
        if len(ids) >= need: break
    return torch.tensor(ids[args.skip * args.T:need], device=dev).view(args.nwin, args.T)


class _Stop(Exception): pass


def linears(blk):
    a, m = blk.self_attn, blk.mlp
    return {"q_proj": a.q_proj, "k_proj": a.k_proj, "v_proj": a.v_proj, "o_proj": a.o_proj,
            "gate_proj": m.gate_proj, "up_proj": m.up_proj, "down_proj": m.down_proj}


@torch.no_grad()
def capture(i, win):
    L = linears(LAYERS[i]); want = {"attn_in": L["q_proj"], "o": L["o_proj"], "mlp_in": L["gate_proj"], "down": L["down_proj"]}
    got = {g: [] for g in want}; hs = []
    for g, mod in want.items():
        hs.append(mod.register_forward_pre_hook(lambda m, a, g=g: got[g].append(a[0].reshape(-1, a[0].shape[-1]).float())))
    hs.append(LAYERS[i].register_forward_hook(lambda *a: (_ for _ in ()).throw(_Stop())))
    for s in win:
        try: model(input_ids=s[None], use_cache=False)
        except _Stop: pass
    for h in hs: h.remove()
    return {g: torch.cat(v) for g, v in got.items()}


def tie_o(s): return s.view(Hkv, REP, hd).mean(1, keepdim=True).expand(Hkv, REP, hd).reshape(-1)


def sq_s(X, Ws, tie):
    ax = X.abs().amax(0).clamp(min=1e-8); aw = torch.stack([W.abs().amax(0) for W in Ws]).amax(0).clamp(min=1e-8)
    s = ax.sqrt() / aw.sqrt(); return tie_o(s) if tie else s


def rotate(A, G):
    N, K = A.shape; return (A.view(N, K // G, G) @ hadamard(G)).view(N, K)


LEARN = torch.load(args.scales, map_location="cpu")["scales"]
win = cal_windows()
layer_ids = range(len(LAYERS)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
res = dict(model=args.tag, hf=args.hf, scales=args.scales, rows=[], sens=[], svec=[])
t0 = time.time()
for i in layer_ids:
    L = linears(LAYERS[i]); X_all = capture(i, win)
    for g, names in GROUPS:
        X = X_all[g]; Ws = [L[n].weight.float() for n in names]
        sl = LEARN[i][g].to(dev).float() if i in LEARN else None
        TF = {"none": torch.ones(X.shape[1], device=dev), "sq": sq_s(X, Ws, g == "o")}
        if sl is not None: TF["learned"] = sl
        # scale-vector analysis (M4-2)
        if sl is not None:
            ls, lq = sl.log(), TF["sq"].log()
            c = torch.corrcoef(torch.stack([ls, lq]))[0, 1].item()
            res["svec"].append(dict(layer=i, group=g, corr_log_learned_sq=c, log_ratio_mean=(ls - lq).mean().item(),
                                    log_ratio_std=(ls - lq).std().item(), learned_log_std=ls.std().item(), sq_log_std=lq.std().item(),
                                    **{f"{k}_{tf}": v for tf in ("none", "sq", "learned") for k, v in block_hetero(X, Ws[0], TF[tf]).items()}))
        for qn, (G, sfmt, f46) in QUANTS.items():
            for tf in list(TF) + [t + "+rot" for t in TF if t in ("none", "learned")]:
                base = tf.split("+")[0]; s = TF[base]; rot = tf.endswith("+rot")
                Xt = X / s; Xt = rotate(Xt, G) if rot else Xt
                ax, Xh = anatomy(Xt, G, sfmt, f46)
                Y = sum(((X @ W.T).pow(2).sum()) for W in Ws)
                oe = 0.0; wrows = []
                for W in Ws:
                    Wt = W * s; Wt = rotate(Wt, G) if rot else Wt
                    aw, Wh = anatomy(Wt, G, sfmt, f46); wrows.append(aw)
                    oe += (Xh @ Wh.T - X @ W.T).pow(2).sum().item()
                    del Wh
                row = dict(layer=i, group=g, quant=qn, tf=tf, out_nmse=oe / Y.item(),
                           **{f"x_{k}": v for k, v in ax.items()},
                           **{f"w_{k}": sum(r[k] for r in wrows) / len(wrows) for k in wrows[0]})
                res["rows"].append(row); del Xh
    del X_all; torch.cuda.empty_cache()
    print(f"[ana {args.tag}] layer {i} anatomy done ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(args.out, "w"))

# ---------------------------------------------------------------- sensitivity: quantize one group only (nvfp4_46 W4A4)
if not args.no_sens:
    @torch.no_grad()
    def nll():
        tot = 0.0
        for s in win:
            tot += F.cross_entropy(model(input_ids=s[None], use_cache=False).logits[0, :-1].float(), s[1:], reduction="sum").item()
        return tot / (win.shape[0] * (win.shape[1] - 1))

    class QG(torch.nn.Module):
        def __init__(self, lin, s, rot):
            super().__init__(); self.lin, self.rot = lin, rot
            W = lin.weight.float() * s; W = rotate(W, 16) if rot else W
            self.w = quant(W, 16, "e4m3", True)[0]; self.s = s
        def forward(self, x):
            sh = x.shape; xs = x.reshape(-1, sh[-1]).float() / self.s; xs = rotate(xs, 16) if self.rot else xs
            y = quant(xs, 16, "e4m3", True)[0] @ self.w.T
            if self.lin.bias is not None: y = y + self.lin.bias.float()
            return y.view(*sh[:-1], -1).to(x.dtype)

    base = nll(); res["base_nll"] = base
    for i in layer_ids:
        L = linears(LAYERS[i]); X_all = capture(i, win[:1])
        for g, names in GROUPS:
            Ws = [L[n].weight.float() for n in names]
            TF = {"none": torch.ones(Ws[0].shape[1], device=dev), "sq": sq_s(X_all[g], Ws, g == "o")}
            if i in LEARN: TF["learned"] = LEARN[i][g].to(dev).float()
            for tf in list(TF) + ["none+rot", "learned+rot"]:
                if tf.split("+")[0] not in TF: continue
                s = TF[tf.split("+")[0]]; rot = tf.endswith("+rot"); saved = {}
                for n in names:
                    parent = LAYERS[i].self_attn if n in ("q_proj", "k_proj", "v_proj", "o_proj") else LAYERS[i].mlp
                    saved[n] = (parent, getattr(parent, n)); setattr(parent, n, QG(getattr(parent, n), s, rot))
                d = nll() - base
                for n, (parent, lin) in saved.items(): setattr(parent, n, lin)
                res["sens"].append(dict(layer=i, group=g, tf=tf, d_nll=d))
        del X_all; torch.cuda.empty_cache()
        print(f"[ana {args.tag}] layer {i} sensitivity done ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open(args.out, "w"))
json.dump(res, open(args.out, "w"))
print("ANA_DONE", flush=True)
