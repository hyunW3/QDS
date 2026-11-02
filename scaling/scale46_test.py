"""Does 4/6 work by fixing the top interval? Direct evidence on every decoder Linear's FPROP / DGRAD / WGRAD.

E2M1, 1x16 blocks, FP32 block scales. For T in {I, H16 (random signs shared by both operands)} and each operand:
  sqnr_6, sqnr_46          tensor SQNR (dB) with absmax -> 6 and with the per-block min-MSE choice of 6 or 4
  frac4                    share of blocks that choose 4
  top_err6, top_err46      share of the operand's squared error (under @6 and under 4/6) carried by elements whose
                           @6-normalized magnitude t = 6|x|/max lies in [4, 6) (the coarsest E2M1 interval)
  top_drop                 relative drop of the squared error of those [4,6) elements, @6 -> 4/6
  nmax4, nmax6             mean number of near-maximal elements (t >= 4, block maximum included) in blocks that choose
                           4 vs blocks that keep 6
GEMM: SQNR of Y for E2M1@6, E2M1-4/6 and the uniform grid (UNIF with H16 = UFP4).
"""
import os
import argparse, json, math, sys
from pathlib import Path
import torch
D = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(D))
from fp4lib import hadamard
from hf_common import load_hf

ap = argparse.ArgumentParser()
ap.add_argument("--hf", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
args = ap.parse_args()
dev = "cuda:0"; G = 16
H = hadamard(G, dev)
gen = torch.Generator(device=dev); gen.manual_seed(1)
SG = (torch.randint(0, 2, (1 << 16,), device=dev, generator=gen) * 2 - 1).float()
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6.], device=dev); TH = (E2M1[1:] + E2M1[:-1]) / 2


def q_at(Mb, amax, m):
    s = amax / m
    return torch.sign(Mb) * E2M1[torch.bucketize((Mb.abs() / s).clamp(max=6), TH)] * s


def analyse(M):
    """M [N, R] -> dequantized @6 and 4/6 plus diagnostics."""
    N, R = M.shape; Mb = M.view(N, R // G, G); amax = Mb.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    Q6, Q4 = q_at(Mb, amax, 6.0), q_at(Mb, amax, 4.0)
    e6, e4 = (Q6 - Mb).pow(2), (Q4 - Mb).pow(2)
    pick4 = e4.sum(-1, keepdim=True) < e6.sum(-1, keepdim=True)
    Q46 = torch.where(pick4, Q4, Q6); e46 = torch.where(pick4, e4, e6)
    t = 6 * Mb.abs() / amax
    top = (t >= 4) & (t < 6 - 1e-6); near = t >= 4 - 1e-6
    sig = Mb.pow(2).sum()
    p4 = pick4.squeeze(-1); nn = near.sum(-1).float()
    d = dict(sqnr_6=-10 * math.log10(e6.sum().item() / sig.item()), sqnr_46=-10 * math.log10(e46.sum().item() / sig.item()),
             frac4=p4.float().mean().item(),
             top_err6=(e6[top].sum() / e6.sum()).item(), top_err46=(e46[top].sum() / e46.sum()).item(),
             top_drop=1 - (e46[top].sum() / e6[top].sum().clamp(min=1e-30)).item(),
             nmax4=nn[p4].mean().item() if p4.any() else float("nan"), nmax6=nn[~p4].mean().item() if (~p4).any() else float("nan"))
    return Q6.view(N, R), Q46.view(N, R), d


def q_unif(M):
    N, R = M.shape; Mb = M.view(N, R // G, G); s = (Mb.abs().amax(-1, keepdim=True) / 7).clamp(min=1e-30)
    return ((Mb / s).round().clamp(-7, 7) * s).view(N, R)


def rot(M):
    return ((M * SG[: M.shape[1]]).view(M.shape[0], -1, G) @ H).view(M.shape)


model, tok = load_hf(args.hf, dev)
ids = []
for line in open(os.environ.get("DCLM_VAL", "data/dclm_val.jsonl")):
    ids += tok.encode(json.loads(line)["text"], add_bos=True, add_eos=True)
    if len(ids) >= 4 * 1025: break
batch = torch.tensor(ids[:4 * 1025], device=dev).view(4, 1025)
fout = open(args.out, "w"); saved = {}


def evaluate(lname, gemm, A, B):
    if A.shape[1] % G: return
    Y = A @ B.T; nY = (Y * Y).sum().item()
    for tn in ("I", "H16"):
        At, Bt = (A, B) if tn == "I" else (rot(A), rot(B))
        A6, A46, da = analyse(At); B6, B46, db_ = analyse(Bt)
        gq = lambda X, Z: -10 * math.log10(((X @ Z.T - Y) ** 2).sum().item() / nY)
        rec = dict(model=args.tag, layer=lname, gemm=gemm, T=tn, A=da, B=db_, gemm_6=gq(A6, B6), gemm_46=gq(A46, B46),
                   gemm_unif=gq(q_unif(At), q_unif(Bt)))
        fout.write(json.dumps(rec) + "\n")


def bwd(name):
    def h(mod, gi, go):
        X = saved.pop(name).reshape(-1, mod.in_features).float()
        dY = go[0].detach().reshape(-1, mod.out_features).float()
        W = mod.weight.detach().float()
        with torch.no_grad():
            evaluate(name, "fprop", X, W)
            evaluate(name, "dgrad", dY, W.T.contiguous())
            evaluate(name, "wgrad", dY.T.contiguous(), X.T.contiguous())
    return h


for n, m in model.named_modules():
    if isinstance(m, torch.nn.Linear) and n.startswith("layers."):
        m.register_forward_hook(lambda mod, i, o, n=n: saved.__setitem__(n, i[0].detach()))
        m.register_full_backward_hook(bwd(n))
loss = model(batch[:, :-1], target=batch[:, 1:]); loss.backward()
print(f"[{args.tag}] loss {loss.item():.4f}", flush=True)
fout.close()
