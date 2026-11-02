"""Q: train in FP4 without any transform, add the transform T only after training (inference)?

Loads Stage-1 checkpoints (e2m1_rtn = trained with plain E2M1 FP4 GEMMs; bf16 = trained in BF16) and evaluates the
validation loss (256 x 1024 tokens, same as training eval) with the FPROP of every decoder Linear in:
  BF16 | E2M1 (as trained by e2m1_rtn) | E2M1-4/6 | UFP4 (uniform + H16) | E2M1 + closed-form smoothing (alpha 0.5,
  calibrated) | E2M1 + learned per-channel diag (calibrated, STE) | E2M1-4/6 + learned diag.
Calibration: 8 x 1024 training tokens (disjoint from validation).
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM
D = Path(__file__).resolve().parent; sys.path.insert(0, str(D.parent))
from fp4lib import E2M1, UNIF, BLK, quantize, hadamard

ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--out", required=True)
args = ap.parse_args()
dev = "cuda"; torch.manual_seed(0)
H = hadamard(BLK, dev)
SG = (torch.randint(0, 2, (1 << 16,), device=dev, generator=torch.Generator(device=dev).manual_seed(0)) * 2 - 1).float()
val = np.memmap(D / "data/val.bin", dtype=np.uint16, mode="r"); train = np.memmap(D / "data/train.bin", dtype=np.uint16, mode="r")
T = 1024
val_x = torch.from_numpy(np.stack([val[i * (T + 1):(i + 1) * (T + 1)].astype(np.int64) for i in range(256)])).to(dev)
cal_x = torch.from_numpy(np.stack([train[i * (T + 1):(i + 1) * (T + 1)].astype(np.int64) for i in range(8)])).to(dev)


def e2m1(A): return quantize(A, E2M1, "rtn", "fp32")[0]
def unif(A): return quantize(A, UNIF, "rtn", "fp32")[0]


def q46(A):
    N, R = A.shape; Ab = A.view(N, R // BLK, BLK)
    amax = Ab.abs().amax(-1, keepdim=True).clamp(min=1e-30); g = E2M1.to(dev); th = (g[1:] + g[:-1]) / 2
    o = []
    for gm in (6.0, 4.0):
        s = amax / gm; q = g[torch.bucketize((Ab.abs() / s).clamp(max=6), th)]; Ah = torch.sign(Ab) * q * s
        o.append((Ah, (Ah - Ab).pow(2).sum(-1, keepdim=True)))
    return torch.where(o[1][1] < o[0][1], o[1][0], o[0][0]).view(N, R)


class QL(torch.nn.Module):
    def __init__(self, lin):
        super().__init__(); self.lin, self.mode, self.cal, self.s = lin, "bf16", [], {}

    def forward(self, x):
        if self.mode == "capture":
            self.cal.append(x.detach().reshape(-1, x.shape[-1]).float().cpu()); return self.lin(x)
        if self.mode == "bf16": return self.lin(x)
        xs = x.reshape(-1, x.shape[-1]).float(); W = self.lin.weight.float()
        grid, tf = self.mode.split("+")
        qf = {"e2m1": e2m1, "unif": unif, "e46": q46}[grid]
        if tf == "h16":
            xs = ((xs * SG[:xs.shape[1]]).view(xs.shape[0], -1, BLK) @ H).view(xs.shape)
            W = ((W * SG[:W.shape[1]]).view(W.shape[0], -1, BLK) @ H).view(W.shape)
        elif tf in ("smooth", "diag"):
            s = self.s[tf]; xs, W = xs / s, W * s
        return (qf(xs) @ qf(W).T).view(*x.shape[:-1], -1).to(x.dtype)


@torch.no_grad()
def val_loss(m):
    tot = 0.0
    for i in range(0, len(val_x), 4):
        b = val_x[i:i + 4]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = m(input_ids=b[:, :-1]).logits
        tot += F.cross_entropy(logits.float().flatten(0, 1), b[:, 1:].flatten()).item() * len(b)
    return tot / len(val_x)


res = {}
for run in args.runs:
    meta = json.load(open(D / "runs" / run / "meta.json"))
    cfg = LlamaConfig(vocab_size=49152, hidden_size=meta["dim"], intermediate_size=meta["ffn"], num_hidden_layers=meta["layers"],
                      num_attention_heads=meta["heads"], num_key_value_heads=meta["heads"], max_position_embeddings=meta["seqlen"],
                      rope_theta=10000.0, tie_word_embeddings=True, attention_bias=False, mlp_bias=False)
    m = LlamaForCausalLM(cfg)
    m.load_state_dict(torch.load(D / "runs" / run / "final.pt", map_location="cpu")); m.to(dev).eval()
    m.requires_grad_(False)
    Q = []
    for layer in m.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for n, c in list(parent.named_children()):
                if isinstance(c, torch.nn.Linear):
                    q = QL(c); setattr(parent, n, q); Q.append(q)
    for q in Q: q.mode = "capture"
    with torch.no_grad():
        for i in range(0, len(cal_x), 4):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                m(input_ids=cal_x[i:i + 4, :-1])
    for q in Q:
        X = torch.cat(q.cal).to(dev); W = q.lin.weight.detach().float(); q.cal = []
        q.s["smooth"] = (X.abs().amax(0).clamp(min=1e-8) ** 0.5) / (W.abs().amax(0).clamp(min=1e-8) ** 0.5)
        Y = X @ W.T; nY = (Y * Y).sum()
        P = torch.log(q.s["smooth"]).detach().clone().requires_grad_(True)              # init at SmoothQuant
        opt = torch.optim.Adam([P], lr=1e-2); best = (float("inf"), None)
        ste = lambda A: A + (e2m1(A.detach()) - A).detach()
        for it in range(250):
            s = torch.exp(P)
            loss = ((ste(X / s) @ ste(W * s).T - Y) ** 2).sum() / nY
            opt.zero_grad(); loss.backward(); opt.step()
            if loss.item() < best[0]: best = (loss.item(), torch.exp(P.detach()))
        q.s["diag"] = best[1]
    r = {}
    for name, mode in [("BF16", "bf16"), ("E2M1", "e2m1+none"), ("E2M1-4/6", "e46+none"), ("UFP4", "unif+h16"),
                       ("E2M1 + smoothing(α=0.5)", "e2m1+smooth"), ("E2M1 + 학습형 대각", "e2m1+diag"), ("E2M1-4/6 + 학습형 대각", "e46+diag")]:
        for q in Q: q.mode = mode
        r[name] = val_loss(m)
        print(f"[{run}] {name:24s} val loss {r[name]:.4f}", flush=True)
    res[run] = r
json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False)
