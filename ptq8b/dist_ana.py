"""Block-normalised magnitude distributions before/after our learned diagonal (and RHT for reference), density- and
energy-weighted, for X and W of an 8B model (CPU).  t = |a| / (block absmax / 6) on 1x16 blocks (E2M1 absmax -> 6 convention).
Same quantity as fig 8 (RHT before/after), now for the diagonal scaling.
  density  : fraction of elements per t-bin (absmax elements, t = 6, reported separately)
  energy   : fraction of a^2 per t-bin (absmax elements separately -- they are exact, zero error)
Also: E2M1 SQNR (fp32 block scale), absmax energy share, crest-factor median, and the E2M1 squared-error share per bin.
"""
import argparse, json, math, os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--hf", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--scales", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--cal", required=True); ap.add_argument("--layers", default="0,8,16,24,-1"); ap.add_argument("--nwin", type=int, default=2)
args = ap.parse_args(); torch.set_num_threads(16)
tok = AutoTokenizer.from_pretrained(args.hf)
model = AutoModelForCausalLM.from_pretrained(args.hf, dtype=torch.bfloat16).eval()
cfg = model.config; Hq, Hkv = cfg.num_attention_heads, cfg.num_key_value_heads; hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // Hq
NL = len(model.model.layers); LAY = sorted({int(x) % NL for x in args.layers.split(",")})
S = torch.load(args.scales, map_location="cpu")["scales"]
E2 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6.]); TH = (E2[1:] + E2[:-1]) / 2
_H = torch.ones(1, 1)
while _H.shape[0] < 16: _H = torch.cat([torch.cat([_H, _H], 1), torch.cat([_H, -_H], 1)], 0)
H16 = _H / 4.0
EDGES = torch.linspace(0, 6, 61)


def stats(A):
    B = A.reshape(-1, 16).float(); B = B[B.pow(2).sum(-1) > 0]
    amax = B.abs().amax(-1, keepdim=True); t = (B.abs() / amax * 6.0)
    isamax = B.abs() == amax
    s = amax / 6.0; Q = torch.sign(B) * E2[torch.bucketize(t.clamp(max=6), TH)] * s; e2 = (Q - B).pow(2); a2 = B.pow(2)
    tt, ia, ee, aa = t.flatten(), isamax.flatten(), e2.flatten(), a2.flatten()
    nonmax = ~ia
    dens = torch.histc(tt[nonmax], 60, 0, 6); ener = torch.zeros(60); errh = torch.zeros(60)
    idx = torch.bucketize(tt[nonmax], EDGES[1:-1])
    ener.index_add_(0, idx, aa[nonmax]); errh.index_add_(0, idx, ee[nonmax])
    crest = (amax.squeeze(-1) / B.pow(2).mean(-1).sqrt())
    return dict(n=int(tt.numel()), dens=dens.tolist(), ener=ener.tolist(), err=errh.tolist(),
                absmax_count=float(ia.float().mean()), absmax_energy=float(aa[ia].sum() / aa.sum()),
                sqnr=float(10 * math.log10(aa.sum() / ee.sum())), crest_p50=float(crest.median()), tot_energy=float(aa.sum()), tot_err=float(ee.sum()))


ids = []
for line in open(args.cal):
    ids += [tok.bos_token_id] * (tok.bos_token_id is not None) + tok.encode(json.loads(line)["text"], add_special_tokens=False)
    if len(ids) >= (600 + args.nwin) * 1024: break
win = torch.tensor(ids[600 * 1024:(600 + args.nwin) * 1024]).view(args.nwin, 1024)
cap = {}
for i in LAY:
    blk = model.model.layers[i]
    for g, m in [("attn_in", blk.self_attn.q_proj), ("o", blk.self_attn.o_proj), ("mlp_in", blk.mlp.gate_proj), ("down", blk.mlp.down_proj)]:
        m.register_forward_pre_hook(lambda mod, a, key=(i, g): cap.setdefault(key, []).append(a[0][0].float()))
with torch.no_grad():
    for s in win: model(input_ids=s[None], use_cache=False)
acc = {}
def add(key, st):
    a = acc.setdefault(key, dict(dens=torch.zeros(60), ener=torch.zeros(60), err=torch.zeros(60), absmax_energy=0.0, tot_energy=0.0, tot_err=0.0, crest=[], n=0, nmax=0.0))
    a["dens"] += torch.tensor(st["dens"]); a["ener"] += torch.tensor(st["ener"]); a["err"] += torch.tensor(st["err"])
    a["absmax_energy"] += st["absmax_energy"] * st["tot_energy"]; a["tot_energy"] += st["tot_energy"]; a["tot_err"] += st["tot_err"]
    a["crest"].append(st["crest_p50"]); a["n"] += st["n"]; a["nmax"] += st["absmax_count"] * st["n"]
for (i, g), xs in sorted(cap.items()):
    blk = model.model.layers[i]
    W = {"attn_in": blk.self_attn.q_proj, "o": blk.self_attn.o_proj, "mlp_in": blk.mlp.gate_proj, "down": blk.mlp.down_proj}[g].weight.float()
    X = torch.cat(xs); s = S[i][g].float()
    for tf in ["none", "diag", "rht"]:
        if tf == "none": Xt, Wt = X, W
        elif tf == "diag": Xt, Wt = X / s, W * s
        else: Xt, Wt = (X.view(X.shape[0], -1, 16) @ H16).reshape(X.shape), (W.view(W.shape[0], -1, 16) @ H16).reshape(W.shape)
        add(("X", tf), stats(Xt)); add(("W", tf), stats(Wt))
    print(f"[dist {args.tag}] layer {i} {g}", flush=True)
out = dict(model=args.tag, layers=LAY, edges=EDGES.tolist(), res={})
for (op, tf), a in acc.items():
    tot_nonmax = float(a["dens"].sum())
    out["res"][f"{op}|{tf}"] = dict(dens=(a["dens"] / a["n"]).tolist(), ener=(a["ener"] / a["tot_energy"]).tolist(), err=(a["err"] / a["tot_err"]).tolist(),
                                    absmax_count=a["nmax"] / a["n"], absmax_energy=a["absmax_energy"] / a["tot_energy"],
                                    sqnr=10 * math.log10(a["tot_energy"] / a["tot_err"]), crest_p50=sum(a["crest"]) / len(a["crest"]))
json.dump(out, open(args.out, "w"), indent=1)
print("DIST_DONE", flush=True)
