"""Can spread outlier energy be made to land in E2M1's dense range [0, 4) instead of the coarse [4, 6)?

E2M1, 1x16 blocks, FP32 block scales, GEMM SQNR (dB) of every decoder Linear's FPROP / DGRAD / WGRAD.
Transform T in {I, H16 (random-sign Hadamard, shared by both operands), OCS (outlier channel splitting: the top-1%
channels by rms(A_k) rms(B_k) have their A column halved and a second copy A_k/2 appended, with B_k duplicated, so
the product is unchanged and the outlier shrinks without mixing), SM (diag(s): SmoothQuant-style per-channel scale
s_k = sqrt(max|A_k| / max|B_k|), A / s and B * s), SMOCS (SM, then OCS on the smoothed operands),
QDS (per-GEMM alpha search of s by quantized output error, see qds()), QDSOCS (QDS, then OCS)}.
Scale rule m = the level the block absmax is mapped to:
  @6 (plain), @4, @3 (fixed: the top levels are left unused, the bulk sees the finer steps),
  @4/6 (per-block min-MSE choice of 6 or 4, Four Over Six),
  @opt (per-block min-MSE over m in {3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8}; m > 6 clips the largest values to 6).
Reference: uniform grid (E1M2 / INT4) @max under I and H16 (= UFP4).
Also logs, per (T, rule), the fraction of block energy whose normalized magnitude falls in [4, 6) before rounding.
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
ap.add_argument("--moe_expert_stride", type=int, default=0, help="tap routed experts e %% stride == 0 (0 = none)")
ap.add_argument("--moe_layer_stride", type=int, default=1)
ap.add_argument("--auto", action="store_true", help="shard the model over all visible GPUs (device_map=auto)")
ap.add_argument("--layers", default=None)
args = ap.parse_args()
dev = "cuda:0"; gen = torch.Generator(device=dev); gen.manual_seed(1)
G = 16
H = hadamard(G, dev)
SG = (torch.randint(0, 2, (1 << 17,), device=dev, generator=gen) * 2 - 1).float()
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6.], device=dev); TH = (E2M1[1:] + E2M1[:-1]) / 2
MS = (3., 3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7., 8.)


def q_at(Ab, m):
    """E2M1 with block absmax mapped to level m (m > 6 clips)."""
    s = (Ab.abs().amax(-1, keepdim=True) / m).clamp(min=1e-30)
    q = E2M1[torch.bucketize((Ab.abs() / s).clamp(max=6), TH)]
    return torch.sign(Ab) * q * s


def q_best(Ab, ms):
    best, be = None, None
    for m in ms:
        Q = q_at(Ab, m); e = (Q - Ab).pow(2).sum(-1, keepdim=True)
        if best is None:
            best, be = Q, e
        else:
            k = e < be; best = torch.where(k, Q, best); be = torch.where(k, e, be)
    return best


def q_unif(Ab):
    s = (Ab.abs().amax(-1, keepdim=True) / 7).clamp(min=1e-30)
    return (Ab / s).round().clamp(-7, 7) * s


RULES = {"@6": lambda Ab: q_at(Ab, 6.), "@4": lambda Ab: q_at(Ab, 4.), "@3": lambda Ab: q_at(Ab, 3.),
         "@4/6": lambda Ab: q_best(Ab, (6., 4.)), "@opt": lambda Ab: q_best(Ab, MS), "UNIF": q_unif}


def bq(M, rule):
    N, R = M.shape
    return RULES[rule](M.view(N, R // G, G)).view(N, R)


def top_share(M):
    """fraction of energy with 6|x|/blockmax in [4, 6) (block max itself excluded: it sits at exactly 6)."""
    Mb = M.view(M.shape[0], -1, G); t = 6 * Mb.abs() / Mb.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    e = Mb.pow(2); return (e[(t >= 4) & (t < 6 - 1e-6)].sum() / e.sum()).item()


def pad16(M):
    r = (-M.shape[1]) % G
    return M if r == 0 else torch.cat([M, M.new_zeros(M.shape[0], r)], 1)


def transforms(A, B):
    R = A.shape[1]
    out = {"I": (A, B)}
    sg = SG[:R]
    out["H16"] = (((A * sg).view(A.shape[0], -1, G) @ H).view(A.shape), ((B * sg).view(B.shape[0], -1, G) @ H).view(B.shape))
    out["OCS"] = ocs(A, B)
    sc = (A.abs().amax(0).clamp(min=1e-12) / B.abs().amax(0).clamp(min=1e-12)).sqrt()
    out["SM"] = (A / sc, B * sc)
    out["SMOCS"] = ocs(*out["SM"])
    out["QDS"] = qds(A, B)
    out["QDSOCS"] = ocs(*out["QDS"])
    return out


def qds(A, B):
    """QDS by alpha search: s = max|A_k|^a / max|B_k|^(1-a), a in {0, .1, ..., 1}, chosen to minimize the E2M1@6 GEMM
    output error (the same quantizer the learned QDS is fitted with; a grid search matches the learned scale)."""
    ra, rb = A.abs().amax(0).clamp(min=1e-12), B.abs().amax(0).clamp(min=1e-12)
    Y = A @ B.T; best = None
    for i in range(11):
        a = i / 10; s = ra.pow(a) / rb.pow(1 - a)
        At, Bt = A / s, B * s
        e = ((bq(At, "@6") @ bq(Bt, "@6").T - Y) ** 2).sum().item()
        if best is None or e < best[0]:
            best = (e, At, Bt)
    return best[1], best[2]


def ocs(A, B):
    R = A.shape[1]
    score = A.pow(2).mean(0).sqrt() * B.pow(2).mean(0).sqrt()
    oc = score.argsort(descending=True)[: max(1, int(math.ceil(0.01 * R)))]
    A2 = A.clone(); A2[:, oc] *= 0.5
    return pad16(torch.cat([A2, A[:, oc] * 0.5], 1)), pad16(torch.cat([B, B[:, oc]], 1))


if args.auto:
    import moe_tap
    ng = torch.cuda.device_count()
    model, tok = moe_tap.load_auto(args.hf, {i: ("10GiB" if i == 0 else "22GiB") for i in range(ng)})
else:
    model, tok = load_hf(args.hf, dev)
ROOT = Path(os.environ.get("LINGUA_ROOT", "lingua_root"))
ids = []
for line in open(os.environ.get("DCLM_VAL", "data/dclm_val.jsonl")):
    ids += tok.encode(json.loads(line)["text"], add_bos=True, add_eos=True)
    if len(ids) >= 4 * 1025: break
batch = torch.tensor(ids[:4 * 1025], device=dev).view(4, 1025)
keep = None if args.layers is None else {int(x) for x in args.layers.split(",")}
fout = open(args.out, "w"); saved = {}


def evaluate(lname, gemm, A, B):
    if A.shape[1] % G or A.shape[0] == 0 or B.shape[0] == 0: return
    Y = A @ B.T; nY = (Y * Y).sum().item()
    if nY == 0: return
    rec = dict(model=args.tag, layer=lname, gemm=gemm, K=A.shape[1])
    for tn, (At, Bt) in transforms(A, B).items():
        rec[f"{tn}|top46"] = 0.5 * (top_share(At) + top_share(Bt))
        for rule in RULES:
            if tn in ("OCS", "SMOCS", "QDSOCS") and rule == "UNIF": continue
            Yh = bq(At, rule) @ bq(Bt, rule).T
            rec[f"{tn}{rule}"] = ((Yh - Y) ** 2).sum().item() / nY
    fout.write(json.dumps(rec) + "\n")


def bwd(name):
    def h(mod, gi, go):
        X = saved.pop(name).reshape(-1, mod.in_features).float().to(dev)
        dY = go[0].detach().reshape(-1, mod.out_features).float().to(dev)
        W = mod.weight.detach().float().to(dev)
        with torch.no_grad():
            evaluate(name, "fprop", X, W)
            evaluate(name, "dgrad", dY, W.T.contiguous())
            evaluate(name, "wgrad", dY.T.contiguous(), X.T.contiguous())
    return h


for n, m in model.named_modules():
    if isinstance(m, torch.nn.Linear) and n.startswith("layers."):
        if keep is not None and int(n.split(".")[1]) not in keep: continue
        m.register_forward_hook(lambda mod, i, o, n=n: saved.__setitem__(n, i[0].detach()))
        m.register_full_backward_hook(bwd(n))
if args.moe_expert_stride:
    import moe_tap
    print("expert modules tapped:", moe_tap.install(model.m, evaluate, dev, args.moe_expert_stride, args.moe_layer_stride), flush=True)
loss = model(batch[:, :-1], target=batch[:, 1:]); loss.backward()
print(f"[{args.tag}] loss {loss.item():.4f}", flush=True)
fout.close()
