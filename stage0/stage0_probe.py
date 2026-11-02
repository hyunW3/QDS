"""Stage 0: tensor-level FP4 shrinkage / SQNR analysis on real lingua llama tensors (arXiv 2606.20381).

For every nn.Linear in the transformer blocks (output head excluded: kept BF16 in FP4 recipes), one
fwd+bwd pass on DCLM val text; in the backward hook we form the three training GEMMs

  FPROP  Y  = X  @ W^T     operands X [N,K],   W   [O,K]   (reduce K)
  DGRAD  dX = dY @ W       operands dY [N,O],  W^T [K,O]   (reduce O)
  WGRAD  dW = dY^T @ X     operands dY^T [O,N], X^T [K,N]  (reduce N)

and for each config (grid x transform x scale fmt x rounding) record per-operand tensor metrics and
GEMM-output metrics vs the exact (fp32 from bf16 operands) product.
"""
import os
import argparse, json, sys, time
from pathlib import Path
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fp4lib import GRIDS, quantize, tensor_metrics, gemm_metrics, hadamard, rht, BLK

ROOT = Path(os.environ.get("LINGUA_ROOT", "lingua_root"))
sys.path.insert(0, str(ROOT / "lingua"))
from lingua.args import dataclass_from_dict
from lingua.tokenizer import build_tokenizer
from apps.main.transformer import LMTransformer, LMTransformerArgs

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=None)
ap.add_argument("--hf", default=None, help="Hugging Face model id (open-source model) instead of --ckpt")
ap.add_argument("--tag", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--seqlen", type=int, default=1024)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tok", default=str(ROOT / "tokenizer/llama3/tokenizer.model"))
ap.add_argument("--val", default=os.environ.get("DCLM_VAL", "data/dclm_val.jsonl"))
args = ap.parse_args()
torch.manual_seed(args.seed)
dev = "cuda"

if args.hf:
    from hf_common import load_hf
    model, tok = load_hf(args.hf, dev)
else:
    p = Path(args.ckpt)
    cfg = OmegaConf.load(p / "params.json")
    model = LMTransformer(dataclass_from_dict(LMTransformerArgs, cfg.model, strict=False))
    st = torch.load(p / "consolidated.pth", weights_only=True, map_location="cpu", mmap=True)
    model.load_state_dict(st["model"]); del st
    model = model.to(dev).train()
    tok = build_tokenizer("tiktoken", args.tok)

# --- data: pack val docs into B x (T+1)
need = args.batch * (args.seqlen + 1)
ids = []
with open(args.val) as f:
    for line in f:
        ids += tok.encode(json.loads(line)["text"], add_bos=True, add_eos=True)
        if len(ids) >= need:
            break
x = torch.tensor(ids[:need], device=dev).view(args.batch, args.seqlen + 1)
inp, tgt = x[:, :-1], x[:, 1:]

# --- configs
# tensor/GEMM configs: grid x transform x scale; rounding of the dY operand varies (paper: SR on dY only)
CFGS = []
for g in GRIDS:
    for tr in ["I", "H"]:
        for sf in ["fp32", "e4m3"]:
            for dy_r in ["rtn", "sr"]:
                CFGS.append(dict(grid=g, tr=tr, scale=sf, dy_round=dy_r))
H = hadamard(BLK, dev)
gen = torch.Generator(device=dev); gen.manual_seed(args.seed + 1)

saved = {}
fout = open(args.out, "w")


def run_gemm(lname, gemm, A, Bm, a_is_dy, b_is_dy):
    """A [M,R], Bm [P,R] (both reduce along last axis). exact product A @ Bm^T."""
    A = A.float(); Bm = Bm.float()
    Y = A @ Bm.T
    signs = (torch.randint(0, 2, (A.shape[1],), device=dev, generator=gen) * 2 - 1).float()
    for c in CFGS:
        if c["dy_round"] == "sr" and not (a_is_dy or b_is_dy):
            continue  # rounding variant only matters for GEMMs that consume dY
        g = GRIDS[c["grid"]]
        At, Bt = (rht(A, signs, H), rht(Bm, signs, H)) if c["tr"] == "H" else (A, Bm)
        rec = dict(model=args.tag, layer=lname, gemm=gemm, **c)
        outs = []
        for role, T, is_dy in [("A", At, a_is_dy), ("B", Bt, b_is_dy)]:
            r = c["dy_round"] if is_dy else "rtn"
            Th, t, q = quantize(T, g, r, c["scale"], gen=gen)
            m = tensor_metrics(T, Th, t, q, g)
            rec[role] = m
            outs.append(Th)
        Yh = outs[0] @ outs[1].T   # H orthogonal & shared -> (AH)(BH)^T = A B^T in exact arithmetic
        rec.update(gemm_metrics(Y, Yh))
        fout.write(json.dumps(rec) + "\n")


def fwd_hook(name):
    def h(mod, inputs, output):
        saved[name] = inputs[0].detach().to(torch.bfloat16)   # autocast feeds bf16 to the GEMM
    return h


def bwd_hook(name):
    def h(mod, grad_in, grad_out):
        X = saved.pop(name).reshape(-1, mod.in_features)
        dY = grad_out[0].detach().reshape(-1, mod.out_features).to(torch.bfloat16)
        W = mod.weight.detach().to(torch.bfloat16)
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            run_gemm(name, "fprop", X, W, False, False)
            run_gemm(name, "dgrad", dY, W.T.contiguous(), True, False)
            run_gemm(name, "wgrad", dY.T.contiguous(), X.T.contiguous(), True, False)
    return h


for n, m in model.named_modules():
    if isinstance(m, torch.nn.Linear) and n.startswith("layers."):
        m.register_forward_hook(fwd_hook(n))
        m.register_full_backward_hook(bwd_hook(n))

t0 = time.time()
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss = model(inp, target=tgt, attn_impl="sdpa")
print(f"[{args.tag}] loss {loss.item():.4f}", flush=True)
loss.backward()
fout.close()
print(f"[{args.tag}] done in {time.time()-t0:.0f}s -> {args.out}", flush=True)
