import json, os, sys, collections
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
dev = "cuda"; hf = sys.argv[1]; sc_path = sys.argv[2]; layers = [int(x) for x in sys.argv[3].split(",")]
tok = AutoTokenizer.from_pretrained(hf); model = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16).to(dev).eval()
E = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6.], device=dev); TH = (E[1:] + E[:-1]) / 2
def q46(A):
    N, K = A.shape; Ab = A.view(N, K // 16, 16); amax = Ab.abs().amax(-1, keepdim=True); gs = (amax.max() / 4 / 448).clamp(min=1e-30); best = None
    for d in (6.0, 4.0):
        s = ((amax / d) / gs).clamp(max=448).to(torch.float8_e4m3fn).float() * gs; s = torch.where(s == 0, torch.ones_like(s), s)
        Ah = torch.sign(Ab) * E[torch.bucketize((Ab.abs() / s).clamp(max=6), TH)] * s; e = (Ah - Ab).pow(2).sum(-1, keepdim=True)
        best = (Ah, e) if best is None else (torch.where(e < best[1], Ah, best[0]), torch.minimum(e, best[1]))
    return best[0].view(N, K)
S = torch.load(sc_path, map_location="cpu")["scales"]
ids = []
for line in open("dclm_val_head2000.jsonl"):
    ids += [tok.bos_token_id] * (tok.bos_token_id is not None) + tok.encode(json.loads(line)["text"], add_special_tokens=False)
    if len(ids) >= 604 * 1024: break
win = torch.tensor(ids[600 * 1024:604 * 1024], device=dev).view(4, 1024)
class _S(Exception): pass
for i in layers:
    blk = model.model.layers[i]
    for g, mod in [("attn_in", blk.self_attn.q_proj), ("mlp_in", blk.mlp.gate_proj)]:
        got = []; h = mod.register_forward_pre_hook(lambda m, a: got.append(a[0][0].float())); h2 = blk.register_forward_hook(lambda *a: (_ for _ in ()).throw(_S()))
        with torch.no_grad():
            for s in win:
                try: model(input_ids=s[None])
                except _S: pass
        h.remove(); h2.remove()
        X = torch.cat(got); s = S[i][g].to(dev).float(); d = S[f"{i}.shift"][g].to(dev).float()
        ea = (q46(X / s) * s - X).pow(2).sum(1); eb = (q46((X - d) / s) * s + d - X).pow(2).sum(1); nx = X.pow(2).sum(1)
        ratio = eb / ea.clamp(min=1e-12); rel_b = eb / nx
        flat = win.flatten(); pos = torch.arange(len(flat), device=dev) % 1024
        top = ratio.topk(8).indices
        print(f"layer {i} {g}: tokens with err x>3: {(ratio > 3).float().mean()*100:.2f}%  x>10: {(ratio > 10).float().mean()*100:.2f}%  | share of total shifted error from top 1% tokens: {(eb.sort(descending=True).values[:len(eb)//100].sum()/eb.sum()).item():.2f}")
        print("   top:", [(int(pos[t]), repr(tok.decode([int(flat[t])])), round(float(ratio[t]), 1), round(float(nx[t].sqrt() / nx.sqrt().median()), 1)) for t in top])
        c = collections.Counter(tok.decode([int(flat[t])]) for t in (ratio > 3).nonzero().flatten().tolist())
        print("   most common tokens among x>3:", c.most_common(6))
