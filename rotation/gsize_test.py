"""E2M1 block statistics of real FPROP operands vs block size G, no rotation vs random-sign block Hadamard (CPU)."""
import os
import json, math, sys, torch
sys.path.insert(0, "."); from fp4lib import hadamard; from hf_common import load_hf
torch.manual_seed(1)
E = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6.]); TH = (E[1:] + E[:-1]) / 2
model, tok = load_hf("Qwen/Qwen2.5-0.5B", "cpu")
ids = []
for line in open(os.environ.get("DCLM_VAL", "data/dclm_val.jsonl")):
    ids += tok.encode(json.loads(line)["text"], add_bos=True, add_eos=True)
    if len(ids) >= 2 * 1025: break
batch = torch.tensor(ids[:2 * 1025]).view(2, 1025)
acc = {}
def stats(M, G, rot):
    N, K = M.shape
    if rot:
        sg = (torch.randint(0, 2, (K,)) * 2 - 1).float(); M = ((M * sg).view(N, -1, G) @ hadamard(G, "cpu")).view(N, K)
    Mb = M.view(N, K // G, G); mx = Mb.abs().amax(-1, keepdim=True).clamp(min=1e-30); e = Mb ** 2
    s = mx / 6; Q = torch.sign(Mb) * E[torch.bucketize((Mb.abs() / s).clamp(max=6), TH)] * s
    err = (Q - Mb) ** 2; t = 6 * Mb.abs() / mx
    return dict(crest=(mx[..., 0] / e.mean(-1).sqrt().clamp(min=1e-30)).median().item(), maxE=(mx ** 2).sum().item() / e.sum().item(),
                nmse=err.sum().item() / e.sum().item(), err_dz=err[t < 0.25].sum().item() / err.sum().item(),
                err_low=err[(t >= 0.25) & (t < 2)].sum().item() / err.sum().item(), err_top=err[(t >= 4) & (t < 6 - 1e-6)].sum().item() / err.sum().item())
def hook(mod, inp, out, n=None):
    X = inp[0].detach().reshape(-1, mod.in_features).float()
    for op, M in (("X", X), ("W", mod.weight.detach().float())):
        for G in (16, 32, 128):
            if M.shape[1] % G: continue
            for rot in (0, 1):
                for k, v in stats(M, G, rot).items(): acc.setdefault((op, G, rot, k), []).append(v)
for n, m in model.named_modules():
    if isinstance(m, torch.nn.Linear) and n.startswith("layers.") and int(n.split(".")[1]) % 4 == 0: m.register_forward_hook(hook)
with torch.no_grad(): model(batch[:, :-1], target=batch[:, 1:])
for op in ("X", "W"):
    for G in (16, 32, 128):
        r = {rot: {k: sum(acc[(op, G, rot, k)]) / len(acc[(op, G, rot, k)]) for k in ("crest", "maxE", "nmse", "err_dz", "err_low", "err_top")} for rot in (0, 1)}
        print(f"{op} G={G:3d} | " + " || ".join(f"{'H' if rot else 'I'}: crest {v['crest']:.2f} maxE {100*v['maxE']:4.1f}% NMSE {10*math.log10(v['nmse']):6.2f}dB err dz/low/top {100*v['err_dz']:4.1f}/{100*v['err_low']:4.1f}/{100*v['err_top']:4.1f}%" for rot, v in r.items()))
