"""Are byte-level models more robust to low-precision GEMMs? BPB degradation under FP8/FP4 forward quantization.

Text: 500-byte windows from DCLM val (the HF BLT port matches official BLT only below 512 bytes), batch size 1.
BPB = sum of next-token cross-entropy (nats) / ln 2 / number of predicted bytes (subword: bytes of predicted tokens).
Quantized: every nn.Linear FPROP (X W^T) except lm_head / embeddings; for BLT also not the entropy model (patcher), so
patch boundaries are unchanged. Formats: FP8 E4M3, FP4 E2M1 RTN, E2M1-4/6, UFP4 (uniform E1M2 + 16x16 RHT); 1x16
blocks along K, FP32 absmax scale. BLT parts: all / local encoder+decoder only / global transformer only.
"""
import os
import argparse, json, math, sys
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
D = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(D))
from fp4lib import E2M1, UNIF, BLK, quantize, hadamard

ap = argparse.ArgumentParser()
ap.add_argument("--hf", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--nwin", type=int, default=200); ap.add_argument("--nbytes", type=int, default=500)
ap.add_argument("--formats", default="fp8,e2m1,e2m1_46,ufp4"); ap.add_argument("--parts", default=None)
args = ap.parse_args()
dev = "cuda"
IS_BLT = "blt" in args.hf.lower()
if IS_BLT:
    m = AutoModelForCausalLM.from_pretrained(args.hf, dtype=torch.bfloat16)
    emb = m.model.encoder_hash_tok_embedding
    m.model.encoder_hash_tok_embedding = None
    m.to(dev); m.model.encoder_hash_tok_embedding = emb          # hash embeddings stay on CPU (lookup only)

    def to_dev_hook(mod, inp, out):
        return out.to(dev) if torch.is_tensor(out) else out
    for e in emb.modules():
        if isinstance(e, torch.nn.Embedding):
            e.register_forward_pre_hook(lambda mod, inp: tuple(i.cpu() if torch.is_tensor(i) else i for i in inp))
            e.register_forward_hook(to_dev_hook)
else:
    m = AutoModelForCausalLM.from_pretrained(args.hf, dtype=torch.bfloat16).to(dev)
m.eval()
tok = AutoTokenizer.from_pretrained(args.hf)
H = hadamard(BLK, dev)
SG = None


def q46(A):
    N, R = A.shape; Ab = A.view(N, R // BLK, BLK)
    amax = Ab.abs().amax(-1, keepdim=True).clamp(min=1e-30); g = E2M1.to(dev); th = (g[1:] + g[:-1]) / 2
    o = []
    for gm in (6.0, 4.0):
        s = amax / gm; q = g[torch.bucketize((Ab.abs() / s).clamp(max=6), th)]; Ah = torch.sign(Ab) * q * s
        o.append((Ah, (Ah - Ab).pow(2).sum(-1, keepdim=True)))
    return torch.where(o[1][1] < o[0][1], o[1][0], o[0][0]).view(N, R)


def fp8(A):
    N, R = A.shape; Ab = A.view(N, R // BLK, BLK)
    s = (Ab.abs().amax(-1, keepdim=True) / 448).clamp(min=1e-30)
    return ((Ab / s).to(torch.float8_e4m3fn).float() * s).view(N, R)


def rht(A):
    global SG
    if SG is None or SG.numel() < A.shape[1]:
        SG = (torch.randint(0, 2, (1 << 16,), device=dev, generator=torch.Generator(device=dev).manual_seed(0)) * 2 - 1).float()
    return ((A * SG[:A.shape[1]]).view(A.shape[0], -1, BLK) @ H).view(A.shape)


FMT = {"fp8": lambda X, W: (fp8(X), fp8(W)), "e2m1": lambda X, W: (quantize(X, E2M1, "rtn", "fp32")[0], quantize(W, E2M1, "rtn", "fp32")[0]),
       "e2m1_46": lambda X, W: (q46(X), q46(W)), "ufp4": lambda X, W: (quantize(rht(X), UNIF, "rtn", "fp32")[0], quantize(rht(W), UNIF, "rtn", "fp32")[0]),
       "e2m1_had": lambda X, W: (quantize(rht(X), E2M1, "rtn", "fp32")[0], quantize(rht(W), E2M1, "rtn", "fp32")[0])}


class QL(torch.nn.Module):
    def __init__(self, lin, part):
        super().__init__(); self.lin, self.part, self.fmt = lin, part, None

    def forward(self, x):
        if self.fmt is None:
            return self.lin(x)
        xs = x.reshape(-1, x.shape[-1]).float(); W = self.lin.weight.float()
        Xq, Wq = FMT[self.fmt](xs, W)
        y = Xq @ Wq.T
        if self.lin.bias is not None: y = y + self.lin.bias.float()
        return y.view(*x.shape[:-1], -1).to(x.dtype)


QS, skipped = [], 0
for name, mod in list(m.named_modules()):
    for cn, child in list(mod.named_children()):
        full = f"{name}.{cn}" if name else cn
        if not isinstance(child, torch.nn.Linear) or "lm_head" in full or "patcher" in full or "embed" in full:
            continue
        if child.in_features % BLK:
            skipped += 1; continue
        part = "local" if ("local_encoder" in full or "local_decoder" in full) else ("global" if "global_transformer" in full else "all")
        q = QL(child, part); setattr(mod, cn, q); QS.append(q)
print(f"[{args.tag}] quantizable Linears {len(QS)} (skipped {skipped}, K not a multiple of 16)", flush=True)

texts = []
for line in open(os.environ.get("DCLM_VAL", "data/dclm_val.jsonl")):
    t = json.loads(line)["text"]
    b = t.encode("utf-8")
    if len(b) < args.nbytes: continue
    t = b[:args.nbytes].decode("utf-8", errors="ignore")
    texts.append(t)
    if len(texts) >= args.nwin: break


@torch.no_grad()
def bpb():
    nats, nbytes = 0.0, 0
    for t in texts:
        if IS_BLT:
            ids = tok(t, return_tensors="pt")["input_ids"].to(dev)              # BOS + bytes (+EOS?)
            ids = ids[:, : 1 + len(t.encode("utf-8"))]
            logits = m(input_ids=ids, use_cache=False).logits.float()
            nats += F.cross_entropy(logits[0, :-1], ids[0, 1:], reduction="sum").item(); nbytes += ids.shape[1] - 1
        else:
            ids = tok(t, add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev)
            logits = m(input_ids=ids, use_cache=False).logits.float()
            nats += F.cross_entropy(logits[0, :-1], ids[0, 1:], reduction="sum").item()
            nbytes += len(tok.decode(ids[0, 1:]).encode("utf-8"))
    return nats / math.log(2) / nbytes


res = dict(model=args.tag, hf=args.hf, n_linear=len(QS))
res["bf16"] = bpb(); print(f"[{args.tag}] bf16 BPB {res['bf16']:.4f}", flush=True)
parts = args.parts.split(",") if args.parts else (["all", "local", "global"] if IS_BLT else ["all"])
for part in parts:
    for f in args.formats.split(","):
        for q in QS:
            q.fmt = f if (part == "all" or q.part == part) else None
        res[f"{part}:{f}"] = bpb()
        print(f"[{args.tag}] {part:6s} {f:8s} BPB {res[f'{part}:{f}']:.4f}  (+{100*(res[f'{part}:{f}']/res['bf16']-1):.2f}%)", flush=True)
json.dump(res, open(args.out, "w"), indent=1)
