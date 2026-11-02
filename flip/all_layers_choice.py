"""All layers: share of blocks picking the 2nd option and MSE change vs E2M1@6 only (from flip_<model>.jsonl).

--q "E2M1|FLIP" -> share of FLIP@6;  --q "4/6" -> share of E2M1@4 (absmax -> 4).
"""
import os
import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--q", default="E2M1|FLIP")
ap.add_argument("--hf", action="store_true", help="use the open-source model runs in hf/")
_a = ap.parse_args(); Q = _a.q
import json, math
from collections import defaultdict
from pathlib import Path

D = Path(__file__).resolve().parents[1]
CK = {"llama100M": os.environ.get("LINGUA_CKPTS", "checkpoints") + "/llama100M",
      "llama300M": os.environ.get("LINGUA_CKPTS", "checkpoints") + "/llama300M",
      "llama760M": os.environ.get("LINGUA_CKPTS", "checkpoints") + "/llama760M"}
N = 4096
db = lambda x: -10 * math.log10(x)
HF = {"qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B", "olmo2-1b": "allenai/OLMo-2-0425-1B", "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B",
      "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B", "qwen3-1.7b": "Qwen/Qwen3-1.7B"}


def hf_shapes(repo, layers):
    """(out, in) of every decoder Linear from the HF config (GQA-aware)."""
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(repo)
    d, ff, H = c.hidden_size, c.intermediate_size, c.num_attention_heads
    kv = getattr(c, "num_key_value_heads", None) or H
    hd = getattr(c, "head_dim", None) or d // H
    S = {"q_proj": (H * hd, d), "k_proj": (kv * hd, d), "v_proj": (kv * hd, d), "o_proj": (d, H * hd),
         "gate_proj": (ff, d), "up_proj": (ff, d), "down_proj": (d, ff)}
    return {ln: S[ln.split(".")[-1]] for ln in layers}


SRC = {m: (D / "hf" / f"flip_{m}.jsonl", repo) for m, repo in HF.items()} if _a.hf else {m: (D / f"flip/flip_{m}.jsonl", ck) for m, ck in CK.items()}
for m, (src, ck) in SRC.items():
    if not src.exists():
        continue
    R = [json.loads(l) for l in open(src)]
    R = [r for r in R if r["q"] in ("E2M1", Q)]
    d = {(r["layer"], r["gemm"], r["tf"], r["q"]): r for r in R}
    layers = sorted({r["layer"] for r in R}, key=lambda s: (int(s.split(".")[1]), s))
    if _a.hf:
        shape = hf_shapes(ck, layers)
    else:
        import torch
        st = torch.load(Path(ck) / "consolidated.pth", map_location="cpu", mmap=True, weights_only=True)["model"]
        shape = {ln: tuple(st[ln + ".weight"].shape) for ln in layers}
        del st
    print(f"\n## {m} ({len(layers)} Linears, {len(layers)//7} layers)\n")
    for tr in ["I", "H16"]:
        tb = {"A": [0, 0], "B": [0, 0]}; acc = []; perlayer = defaultdict(list)
        for ln in layers:
            O, K = shape[ln]
            sz = {"fprop": (N * K, O * K), "dgrad": (N * O, K * O), "wgrad": (O * N, K * N)}
            for g in ["fprop", "dgrad", "wgrad"]:
                a, b = d[(ln, g, tr, "E2M1")], d[(ln, g, tr, Q)]
                for k, sh, n in [("A", b["share_A"][1], sz[g][0] / 16), ("B", b["share_B"][1], sz[g][1] / 16)]:
                    tb[k][0] += n; tb[k][1] += sh * n
                acc.append((g, a["g_nmse"], b["g_nmse"], a["A_nmse"], b["A_nmse"]))
                perlayer[int(ln.split(".")[1])].append((a["g_nmse"], b["g_nmse"]))
        print(f"### {tr}: {Q} 2nd-option block share A {100*tb['A'][1]/tb['A'][0]:.1f}%, B {100*tb['B'][1]/tb['B'][0]:.1f}%, all {100*(tb['A'][1]+tb['B'][1])/(tb['A'][0]+tb['B'][0]):.1f}%")
        for g in ["fprop", "dgrad", "wgrad", None]:
            xs = [x for x in acc if g is None or x[0] == g]
            mm = lambda i: sum(x[i] for x in xs) / len(xs)
            print(f"- {g or 'all'}: GEMM NMSE {mm(1)*1e3:.2f} → {mm(2)*1e3:.2f} ({100*(mm(2)/mm(1)-1):+.1f}%; {db(mm(1)):.2f} → {db(mm(2)):.2f} dB), A operand {mm(3)*1e3:.2f} → {mm(4)*1e3:.2f} ({100*(mm(4)/mm(3)-1):+.1f}%)")
        ch = {L: 100 * (sum(y for _, y in v) / sum(x for x, _ in v) - 1) for L, v in perlayer.items()}
        print("- per layer GEMM NMSE change: " + ", ".join(f"L{L} {c:+.1f}%" for L, c in sorted(ch.items())))
