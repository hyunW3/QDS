"""Flipped E2M1 ("FLIP"): quantize u = absmax - |x| with E2M1, i.e. magnitude levels absmax - q*s.

Every option is a magnitude level set in units of the block absmax (t = |x| / absmax in [0, 1]):
  E2M1@6  E2M1/6                         (standard NVFP4: absmax -> 6)
  E2M1@4  E2M1[<=4]/4                    (absmax -> 4, Four-Over-Six's second option)
  FLIP@6  1 - E2M1/6  = {0,2,3,4,4.5,5,5.5,6}/6   (dense near absmax, coarse near 0)
  FLIP@4  1 - E2M1[<=4]/4
  UNIF    arange(8)/7                    (E1M2 = INT4-sym)
Quantizers are single options or per-block min-MSE choices among several (choice stored in the block scale /
a per-block flag). Hardware note: FLIP blocks decode to sign*absmax - sign*s*q = difference of two NVFP4
tensors (codes +-1 with scale absmax, and sign*q with scale s), i.e. two FP4 GEMMs for that operand.

Transforms: I and H16 (16x16 RHT with random signs). FPROP / DGRAD / WGRAD GEMM-output SQNR on one eval batch.
"""
import os
import argparse, json, math, sys
from pathlib import Path
import torch
from omegaconf import OmegaConf

D = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(D))
from fp4lib import E2M1, BLK, hadamard
ROOT = Path(os.environ.get("LINGUA_ROOT", "lingua_root")); sys.path.insert(0, str(ROOT / "lingua"))
from lingua.args import dataclass_from_dict
from lingua.tokenizer import build_tokenizer
from apps.main.transformer import LMTransformer, LMTransformerArgs

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=None); ap.add_argument("--hf", default=None); ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--only", default="", help="comma list of quantizer names to run (default: all)")
args = ap.parse_args()
ONLY = set(args.only.split(",")) if args.only else None
dev = "cuda"
gen = torch.Generator(device=dev); gen.manual_seed(1)

e = E2M1.to(dev)
L = {
    "E2M1@6": e / 6,
    "E2M1@4": e[e <= 4] / 4,
    "FLIP@6": torch.sort(1 - e / 6).values,
    "FLIP@4": torch.sort(1 - e[e <= 4] / 4).values,
    "UNIF": torch.arange(8, device=dev).float() / 7,
}
QUANTS = {
    "E2M1": ["E2M1@6"],
    "FLIP": ["FLIP@6"],
    "UNIF": ["UNIF"],
    "E2M1|FLIP": ["E2M1@6", "FLIP@6"],
    "4/6": ["E2M1@6", "E2M1@4"],
    "4/6|FLIP": ["E2M1@6", "E2M1@4", "FLIP@6"],
    "4/6|FLIP4/6": ["E2M1@6", "E2M1@4", "FLIP@6", "FLIP@4"],
}
# asymmetric: FLIP only on operand A (activations / gradients), 4/6 on operand B -> one extra FP4 GEMM
QUANTS_AB = {"4/6|FLIP4/6 (A only)": (["E2M1@6", "E2M1@4", "FLIP@6", "FLIP@4"], ["E2M1@6", "E2M1@4"])}


def quant(A, opts):
    """per-block min-MSE choice among level sets; returns (A_hat, share of blocks per option)."""
    N, R = A.shape
    Ab = A.float().view(N, R // BLK, BLK)
    amax = Ab.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    t = Ab.abs() / amax
    best = berr = bidx = None
    for i, o in enumerate(opts):
        lv = L[o]; th = (lv[1:] + lv[:-1]) / 2
        Ah = torch.sign(Ab) * lv[torch.bucketize(t, th)] * amax
        err = (Ah - Ab).pow(2).sum(-1, keepdim=True)
        if best is None:
            best, berr, bidx = Ah, err, torch.zeros_like(err, dtype=torch.long)
        else:
            m = err < berr
            best = torch.where(m, Ah, best); berr = torch.where(m, err, berr); bidx = torch.where(m, i, bidx)
    share = [(bidx == i).float().mean().item() for i in range(len(opts))]
    return best.view(N, R), share


H = hadamard(BLK, dev)


def rht(M, sg):
    N, R = M.shape
    return ((M * sg).view(N, R // BLK, BLK) @ H).view(N, R)


if args.hf:
    from hf_common import load_hf
    model, tok = load_hf(args.hf, dev)
else:
    ck = Path(args.ckpt)
    cfg = OmegaConf.load(ck / "params.json")
    model = LMTransformer(dataclass_from_dict(LMTransformerArgs, cfg.model, strict=False))
    model.load_state_dict(torch.load(ck / "consolidated.pth", weights_only=True, map_location="cpu", mmap=True)["model"])
    model = model.to(dev).train()
    tok = build_tokenizer("tiktoken", str(ROOT / "tokenizer/llama3/tokenizer.model"))
ids = []
for line in open(os.environ.get("DCLM_VAL", "data/dclm_val.jsonl")):
    ids += tok.encode(json.loads(line)["text"], add_bos=True, add_eos=True)
    if len(ids) >= 8 * 1025: break
x = torch.tensor(ids[4 * 1025:8 * 1025], device=dev).view(4, 1025)   # same eval batch as rht_variants
saved = {}; fout = open(args.out, "w")


def evaluate(lname, gemm, A, B):
    Y = A @ B.T; nY = (Y * Y).sum()
    sg = (torch.randint(0, 2, (A.shape[1],), device=dev, generator=gen) * 2 - 1).float()
    for tr in ["I", "H16"]:
        At, Bt = (rht(A, sg), rht(B, sg)) if tr == "H16" else (A, B)
        for qn, (oa, ob) in [(k, (v, v)) for k, v in QUANTS.items()] + list(QUANTS_AB.items()):
            if ONLY and qn not in ONLY:
                continue
            opts = oa
            Ah, shA = quant(At, oa); Bh, shB = quant(Bt, ob)
            Yh = Ah @ Bh.T
            fout.write(json.dumps(dict(model=args.tag, layer=lname, gemm=gemm, tf=tr, q=qn, opts=opts,
                                       g_nmse=(((Yh - Y) ** 2).sum() / nY).item(),
                                       A_nmse=(((Ah - At) ** 2).sum() / (At * At).sum()).item(),
                                       share_A=shA, share_B=shB)) + "\n")


def bwd(name):
    def h(mod, gi, go):
        X = saved.pop(name).reshape(-1, mod.in_features).float()
        dY = go[0].detach().reshape(-1, mod.out_features).to(torch.bfloat16).float()
        W = mod.weight.detach().to(torch.bfloat16).float()
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            evaluate(name, "fprop", X, W)
            evaluate(name, "dgrad", dY, W.T.contiguous())
            evaluate(name, "wgrad", dY.T.contiguous(), X.T.contiguous())
    return h


for n, m in model.named_modules():
    if isinstance(m, torch.nn.Linear) and n.startswith("layers."):
        m.register_forward_hook(lambda mod, i, o, n=n: saved.__setitem__(n, i[0].detach().to(torch.bfloat16)))
        m.register_full_backward_hook(bwd(n))
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss = model(x[:, :-1], target=x[:, 1:], attn_impl="sdpa")
loss.backward(); fout.close()
print(f"[{args.tag}] loss {loss.item():.4f} done -> {args.out}", flush=True)
