"""Direction 3: learn an E2M1-aware invertible transform per Linear and GEMM (Qwen2.5-0.5B, layers 0/8/16).

For a GEMM Y = A B^T reducing over R, a transform is applied per 16-block of the reduction axis:
    A' = A * blockdiag(M),  B' = B * blockdiag(M^{-T})   (product unchanged)
and trained (Adam, straight-through estimator through the quantizer) to minimize the E2M1 GEMM-output error
||Q(A') Q(B')^T - Y||^2 / ||Y||^2 on a calibration batch; evaluated on a different batch.
Parametrizations:
  ORTH-I     M = Cayley(S), S skew, starts at I            (one 16x16 shared by all blocks)
  ORTH-H     M = H16 * Cayley(S), starts at the Hadamard
  INV        M = I + U (general invertible), starts at I
  GROUP-DIAG per-16-group diagonal (learned smoothing)       (channel axes only)
  GROUP-INV  per-16-group I + U_g                            (channel axes only)
Baselines (no learning): I, H16, uniform grid + H16 (UFP4).
"""
import os
import json, math, sys
from pathlib import Path
import torch
D = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(D))
from fp4lib import E2M1, UNIF, BLK, quantize, hadamard
from hf_common import load_hf

dev = "cuda"; torch.manual_seed(0)
H = hadamard(BLK, dev)
model, tok = load_hf("Qwen/Qwen2.5-0.5B", dev)
LAYERS = {0, 8, 16}
ROOT = Path(os.environ.get("LINGUA_ROOT", "lingua_root"))
ids = []
for line in open(os.environ.get("DCLM_VAL", "data/dclm_val.jsonl")):
    ids += tok.encode(json.loads(line)["text"], add_bos=True, add_eos=True)
    if len(ids) >= 4 * 1025: break
batches = torch.tensor(ids[:4 * 1025], device=dev).view(2, 2, 1025)        # calib, eval: 2048 tokens each
cap = {}


def hook_f(n):
    return lambda mod, i, o: cap.setdefault(n, {}).setdefault("X", []).append(i[0].detach().reshape(-1, mod.in_features).float().cpu())


def hook_b(n):
    return lambda mod, gi, go: cap.setdefault(n, {}).setdefault("dY", []).append(go[0].detach().reshape(-1, mod.out_features).float().cpu())


mods = {}
for n, m in model.named_modules():
    if isinstance(m, torch.nn.Linear) and n.startswith("layers.") and int(n.split(".")[1]) in LAYERS:
        mods[n] = m; m.register_forward_hook(hook_f(n)); m.register_full_backward_hook(hook_b(n))
for b in batches:
    loss = model(b[:, :-1], target=b[:, 1:]); loss.backward()
del model; torch.cuda.empty_cache()


def q_ste(A, grid=E2M1):
    Ah = quantize(A.detach(), grid, "rtn", "fp32")[0]
    return A + (Ah - A).detach()


def q46(A):
    N, R = A.shape; Ab = A.view(N, R // BLK, BLK)
    amax = Ab.abs().amax(-1, keepdim=True).clamp(min=1e-30); g = E2M1.to(dev); th = (g[1:] + g[:-1]) / 2
    o = []
    for gm in (6.0, 4.0):
        s = amax / gm; q = g[torch.bucketize((Ab.abs() / s).clamp(max=6), th)]; Ah = torch.sign(Ab) * q * s
        o.append((Ah, (Ah - Ab).pow(2).sum(-1, keepdim=True)))
    return torch.where(o[1][1] < o[0][1], o[1][0], o[0][0]).view(N, R)


def tf_pair(kind, P, G):
    eye = torch.eye(BLK, device=dev)
    if kind in ("ORTH-I", "ORTH-H"):
        S = P - P.transpose(-1, -2)
        C = torch.linalg.solve(eye + S, eye - S)              # Cayley, orthogonal
        M = H @ C if kind == "ORTH-H" else C
        return M, M                                          # M^{-T} = M
    if kind == "INV":
        M = eye + P
        return M, torch.linalg.inv(M).transpose(-1, -2)
    if kind == "GROUP-DIAG":
        M = torch.diag_embed(torch.exp(P))                   # [G,16,16]
        return M, torch.diag_embed(torch.exp(-P))
    if kind == "GROUP-INV":
        M = eye + P
        return M, torch.linalg.inv(M).transpose(-1, -2)


def blockmul(A, M):
    N, R = A.shape
    if M.dim() == 2:
        return (A.view(N, R // BLK, BLK) @ M).view(N, R)
    return (A.view(N, R // BLK, 1, BLK) @ M.unsqueeze(0)).view(N, R)


def gemm_err(A, B, M, Mi, qf):
    Y = A @ B.T
    Yh = qf(blockmul(A, M)) @ qf(blockmul(B, Mi)).T
    return ((Yh - Y) ** 2).sum() / (Y ** 2).sum()


def fit(kind, A, B, steps=300):
    R = A.shape[1]; G = R // BLK
    shape = (G, BLK) if kind == "GROUP-DIAG" else ((G, BLK, BLK) if kind == "GROUP-INV" else (BLK, BLK))
    P = torch.zeros(shape, device=dev, requires_grad=True)
    opt = torch.optim.Adam([P], lr=1e-2 if kind.startswith("ORTH") or kind == "GROUP-DIAG" else 3e-3)
    best = (float("inf"), None)
    for it in range(steps):
        M, Mi = tf_pair(kind, P, G)
        loss = gemm_err(A, B, M, Mi, q_ste)
        opt.zero_grad(); loss.backward(); opt.step()
        if loss.item() < best[0]:
            best = (loss.item(), P.detach().clone())
    return best[1]


db = lambda x: -10 * math.log10(max(x, 1e-30))
KINDS = ["ORTH-I", "ORTH-H", "INV", "GROUP-DIAG", "GROUP-INV"]
out = []
eye = torch.eye(BLK, device=dev)
for n, c in cap.items():
    Xc, Xe = (t.to(dev) for t in c["X"])                     # one forward+backward per batch: list order = batch order
    dYc, dYe = (t.to(dev) for t in c["dY"])
    W = mods[n].weight.detach().float()
    for gemm, (Ac, Bc, Ae, Be, chan) in {"fprop": (Xc, W, Xe, W, True), "dgrad": (dYc, W.T.contiguous(), dYe, W.T.contiguous(), True),
                                        "wgrad": (dYc.T.contiguous(), Xc.T.contiguous(), dYe.T.contiguous(), Xe.T.contiguous(), False)}.items():
        rec = dict(layer=n, gemm=gemm)
        with torch.no_grad():
            rec["I"] = db(gemm_err(Ae, Be, eye, eye, lambda A: quantize(A, E2M1, "rtn", "fp32")[0]))
            rec["I_46"] = db(gemm_err(Ae, Be, eye, eye, q46))
            rec["H16"] = db(gemm_err(Ae, Be, H, H, lambda A: quantize(A, E2M1, "rtn", "fp32")[0]))
            rec["UFP4"] = db(gemm_err(Ae, Be, H, H, lambda A: quantize(A, UNIF, "rtn", "fp32")[0]))
        for kind in KINDS:
            if kind.startswith("GROUP") and not chan:
                continue
            P = fit(kind, Ac, Bc)
            with torch.no_grad():
                M, Mi = tf_pair(kind, P, Ac.shape[1] // BLK)
                rec[kind] = db(gemm_err(Ae, Be, M, Mi, lambda A: quantize(A, E2M1, "rtn", "fp32")[0]))
                rec[kind + "_46"] = db(gemm_err(Ae, Be, M, Mi, q46))
                rec[kind + "_calib"] = db(gemm_err(Ac, Bc, M, Mi, lambda A: quantize(A, E2M1, "rtn", "fp32")[0]))
        out.append(rec)
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
json.dump(out, open(D / "hf/learned_tf_qwen2.5-0.5b.json", "w"), indent=1)
