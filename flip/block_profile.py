"""Which blocks pick E2M1@6 / E2M1@4 / FLIP@6? Block-shape profile on real tensors (llama 300M, one layer, H16).

For every 1x16 block (after 16x16 RHT) the min-MSE option among {E2M1@6, E2M1@4, FLIP@6} is chosen; blocks are then
profiled by t = |x|/absmax: share of small elements (t < 1/4), share of large elements (t > 2/3), and the MSE gain of
the chosen option over E2M1@6.
"""
import os
import json, sys
from pathlib import Path
import torch
from omegaconf import OmegaConf
D = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(D))
from fp4lib import E2M1, BLK, hadamard
ROOT = Path(os.environ.get("LINGUA_ROOT", "lingua_root")); sys.path.insert(0, str(ROOT / "lingua"))
from lingua.args import dataclass_from_dict
from lingua.tokenizer import build_tokenizer
from apps.main.transformer import LMTransformer, LMTransformerArgs
dev = "cuda"; torch.manual_seed(0)
e = E2M1.to(dev)
L = {"E2M1@6": e / 6, "E2M1@4": e[e <= 4] / 4, "FLIP@6": torch.sort(1 - e / 6).values}
ck = Path(os.environ.get("LINGUA_CKPTS", "checkpoints")) / "llama300M"
cfg = OmegaConf.load(ck / "params.json")
model = LMTransformer(dataclass_from_dict(LMTransformerArgs, cfg.model, strict=False))
model.load_state_dict(torch.load(ck / "consolidated.pth", weights_only=True, map_location="cpu", mmap=True)["model"])
model = model.to(dev).train()
tok = build_tokenizer("tiktoken", str(ROOT / "tokenizer/llama3/tokenizer.model"))
ids = []
for line in open(os.environ.get("DCLM_VAL", "data/dclm_val.jsonl")):
    ids += tok.encode(json.loads(line)["text"], add_bos=True, add_eos=True)
    if len(ids) >= 4 * 1025: break
x = torch.tensor(ids[:4 * 1025], device=dev).view(4, 1025)
cap = {}
TARGETS = ["layers.7.feed_forward.w2", "layers.7.attention.wq"]
for n, m in model.named_modules():
    if n in TARGETS:
        m.register_forward_hook(lambda mod, i, o, n=n: cap.__setitem__((n, "X"), i[0].detach().float().reshape(-1, mod.in_features)))
        m.register_full_backward_hook(lambda mod, gi, go, n=n: cap.__setitem__((n, "dY"), go[0].detach().float().reshape(-1, mod.out_features)))
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss = model(x[:, :-1], target=x[:, 1:], attn_impl="sdpa")
loss.backward()
H = hadamard(BLK, dev)
for (n, kind), A in cap.items():
    for opn, T in ([("fwd_x", A)] if kind == "X" else [("dgrad_dy", A), ("wgrad_dyT", A.T.contiguous())]):
        sg = (torch.randint(0, 2, (T.shape[1],), device=dev) * 2 - 1).float()
        Tb = ((T * sg).view(T.shape[0], -1, BLK) @ H).reshape(-1, BLK)
        amax = Tb.abs().amax(-1, keepdim=True).clamp(min=1e-30); t = Tb.abs() / amax
        errs = []
        for o, lv in L.items():
            q = lv[torch.bucketize(t, (lv[1:] + lv[:-1]) / 2)]
            errs.append(((q - t) ** 2).sum(-1))
        E = torch.stack(errs, -1); ch = E.argmin(-1)
        print(f"\n{n} {opn} (H16): {Tb.shape[0]} blocks")
        print("  option  | share  | mean #small (t<1/4) | mean #large (t>2/3) | MSE gain vs @6")
        for i, o in enumerate(L):
            msk = ch == i
            if msk.sum() == 0: continue
            g = 1 - (E[msk, i].sum() / E[msk, 0].sum()).item()
            print(f"  {o:7s} | {100*msk.float().mean().item():5.1f}% | {(t[msk] < 0.25).float().sum(-1).mean().item():6.2f} / 16 | {(t[msk] > 2/3).float().sum(-1).mean().item():6.2f} / 16 | {100*g:5.1f}%")
