"""Tensor-level rotation gain on 8B models vs block size: FPROP GEMM SQNR of E2M1 (FP32 block scales, absmax -> 6) with and
without a randomized block Hadamard (random signs + H_G), G in {16, 32, 128}, every 4th decoder layer, 2 x 1024 DCLM tokens.
Also logs the activation crest factor and the activation dead-zone / [4,6) error shares (as in gsize_test.py)."""
import argparse, json, math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--hf", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--cal", default="dclm_val_head2000.jsonl"); ap.add_argument("--scale", default="fp32", choices=["fp32", "e4m3"])
ap.add_argument("--stride", type=int, default=4); ap.add_argument("--gs", default="16,32,128")
args = ap.parse_args()
dev = "cuda"
E = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6.], device=dev); TH = (E[1:] + E[:-1]) / 2


def had(G):
    H = torch.ones(1, 1, device=dev)
    while H.shape[0] < G: H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(G)


GS = tuple(int(g) for g in args.gs.split(","))
HS = {g: had(g) for g in GS}


def q(M, G):
    N, K = M.shape; Mb = M.view(N, K // G, G); mx = Mb.abs().amax(-1, keepdim=True).clamp(min=1e-30); s = mx / 6
    if args.scale == "e4m3":                       # NVFP4: E4M3 block scale times an FP32 tensor scale (as in ptq8b.py)
        gs = (mx.max() / 6 / 448).clamp(min=1e-30); s = ((s / gs).clamp(max=448).to(torch.float8_e4m3fn).float() * gs)
        s = torch.where(s == 0, torch.ones_like(s), s)
    return (torch.sign(Mb) * E[torch.bucketize((Mb.abs() / s).clamp(max=6), TH)] * s).view(N, K), (6 * Mb.abs() / mx).view(N, K), mx


tok = AutoTokenizer.from_pretrained(args.hf)
model = AutoModelForCausalLM.from_pretrained(args.hf, dtype=torch.bfloat16).to(dev).eval()
ids = []
for line in open(args.cal):
    ids += [tok.bos_token_id] * (tok.bos_token_id is not None) + tok.encode(json.loads(line)["text"], add_special_tokens=False)
    if len(ids) >= 2 * 1024: break
batch = torch.tensor(ids[:2 * 1024], device=dev).view(2, 1024)
fout = open(args.out, "w")


def hook(name):
    def h(mod, inp, out):
        X = inp[0].detach().reshape(-1, mod.in_features).float(); W = mod.weight.detach().float()
        Y = X @ W.T; nY = (Y * Y).sum().item(); K = X.shape[1]
        g0 = torch.Generator(device="cpu"); g0.manual_seed(1234 + K)
        sg = (torch.randint(0, 2, (K,), generator=g0) * 2 - 1).float().to(dev)
        rec = dict(model=args.tag, layer=name)
        for G in GS:
            for T in ("I", "R"):
                A, B = (X, W) if T == "I" else (((X * sg).view(-1, K // G, G) @ HS[G]).view(X.shape), ((W * sg).view(-1, K // G, G) @ HS[G]).view(W.shape))
                Aq, tA, mx = q(A, G); Bq, _, _ = q(B, G)
                eA = (Aq - A) ** 2
                rec[f"{T}{G}"] = dict(sqnr=-10 * math.log10(((Aq @ Bq.T - Y) ** 2).sum().item() / nY),
                                      crest=(mx[..., 0] / A.view(A.shape[0], -1, G).pow(2).mean(-1).sqrt().clamp(min=1e-30)).median().item(),
                                      dz=eA[tA < 0.25].sum().item() / eA.sum().item(), top=eA[(tA >= 4) & (tA < 6 - 1e-6)].sum().item() / eA.sum().item())
        fout.write(json.dumps(rec) + "\n")
    return h


for n, m in model.named_modules():
    if isinstance(m, torch.nn.Linear) and ".layers." in n and int(n.split(".layers.")[1].split(".")[0]) % args.stride == 0:
        m.register_forward_hook(hook(n))
with torch.no_grad():
    model(input_ids=batch, use_cache=False)
fout.close()
R = [json.loads(l) for l in open(args.out)]
for G in GS:
    m = lambda T, k: sum(r[f"{T}{G}"][k] for r in R) / len(R)
    print(f"[{args.tag}] G={G:3d} rotation gain {m('R','sqnr') - m('I','sqnr'):+.2f} dB | act crest I {m('I','crest'):.2f} R {m('R','crest'):.2f} | "
          f"act err dead-zone I {100*m('I','dz'):.0f}% R {100*m('R','dz'):.0f}% | [4,6) I {100*m('I','top'):.0f}% R {100*m('R','top'):.0f}%", flush=True)
