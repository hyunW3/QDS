"""Two options only: E2M1@6 vs FLIP@6. Which blocks pick FLIP? (llama 300M, layers 0/7/14, I and H16)

Per 1x16 block (t = |x|/absmax): min-MSE choice between E2M1@6 and FLIP@6. Profile the chosen groups and the
probability of picking FLIP as a function of how many elements are near absmax (t > 2/3) and near zero (t < 1/4).
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
L = {"E2M1@6": e / 6, "FLIP@6": torch.sort(1 - e / 6).values}
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
TARGETS = [f"layers.{i}.{m}" for i in (0, 7, 14) for m in ("attention.wq", "feed_forward.w2")]
for n, m in model.named_modules():
    if n in TARGETS:
        m.register_forward_hook(lambda mod, i, o, n=n: cap.__setitem__((n, "X"), i[0].detach().float().reshape(-1, mod.in_features)))
        m.register_full_backward_hook(lambda mod, gi, go, n=n: cap.__setitem__((n, "dY"), go[0].detach().float().reshape(-1, mod.out_features)))
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss = model(x[:, :-1], target=x[:, 1:], attn_impl="sdpa")
loss.backward()
H = hadamard(BLK, dev)
res = []
for (n, kind), A in sorted(cap.items()):
    ops = [("fwd_x", A)] if kind == "X" else [("dgrad_dy", A), ("wgrad_dyT", A.T.contiguous())]
    for opn, T in ops:
        sg = (torch.randint(0, 2, (T.shape[1],), device=dev) * 2 - 1).float()
        for tr in ["I", "H16"]:
            Tb = ((T * sg).view(T.shape[0], -1, BLK) @ H).reshape(-1, BLK) if tr == "H16" else T.reshape(-1, BLK)
            amax = Tb.abs().amax(-1, keepdim=True).clamp(min=1e-30); t = Tb.abs() / amax
            E = torch.stack([((lv[torch.bucketize(t, (lv[1:] + lv[:-1]) / 2)] - t) ** 2).sum(-1) for lv in L.values()], -1)
            flip = E[:, 1] < E[:, 0]
            nlarge = (t > 2 / 3).sum(-1); nsmall = (t < 0.25).sum(-1)
            rec = dict(layer=n, op=opn, tr=tr, blocks=int(Tb.shape[0]), flip_share=flip.float().mean().item(),
                       gain_flip_blocks=1 - (E[flip, 1].sum() / E[flip, 0].sum()).item() if flip.any() else 0.0,
                       gain_total=1 - (torch.minimum(E[:, 0], E[:, 1]).sum() / E[:, 0].sum()).item())
            for nm, msk in [("E2M1", ~flip), ("FLIP", flip)]:
                rec[f"{nm}_large"] = nlarge[msk].float().mean().item() if msk.any() else None
                rec[f"{nm}_small"] = nsmall[msk].float().mean().item() if msk.any() else None
            rec["p_flip_by_large"] = [flip[nlarge == k].float().mean().item() if (nlarge == k).any() else None for k in range(17)]
            rec["n_by_large"] = [int((nlarge == k).sum()) for k in range(17)]
            rec["p_flip_by_small"] = [flip[nsmall == k].float().mean().item() if (nsmall == k).any() else None for k in range(17)]
            res.append(rec)
json.dump(res, open(D / "flip/block_profile2.json", "w"), indent=1)
print("| layer | op | T | FLIP 블록 비율 | E2M1 블록: 큰/작은 원소 | FLIP 블록: 큰/작은 원소 | FLIP 블록 MSE 감소 | 텐서 전체 MSE 감소 |")
print("|---|---|---|---|---|---|---|---|")
for r in res:
    f = lambda v: "–" if v is None else f"{v:.2f}"
    print(f"| {r['layer'].replace('layers.','L').replace('attention.','').replace('feed_forward.','')} | {r['op']} | {r['tr']} | {100*r['flip_share']:.1f}% | "
          f"{f(r['E2M1_large'])} / {f(r['E2M1_small'])} | {f(r['FLIP_large'])} / {f(r['FLIP_small'])} | {100*r['gain_flip_blocks']:.1f}% | {100*r['gain_total']:.1f}% |")
