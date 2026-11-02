"""Layer 7 (llama 300M): FLIP@6 block share and MSE change, E2M1@6 only vs per-block E2M1@6|FLIP@6 (from flip_llama300M.jsonl)."""
import json, math
from pathlib import Path
D = Path(__file__).resolve().parents[1]
R = [json.loads(l) for l in open(D / "flip/flip_llama300M.jsonl")]
R = [r for r in R if r["layer"].startswith("layers.7.") and r["q"] in ("E2M1", "E2M1|FLIP")]
d = {(r["layer"].split(".")[-1], r["gemm"], r["tf"], r["q"]): r for r in R}
N, dim, ffn = 4096, 1280, 3584
shape = {"wq": (dim, dim), "wk": (dim, dim), "wv": (dim, dim), "wo": (dim, dim), "w1": (ffn, dim), "w3": (ffn, dim), "w2": (dim, ffn)}
db = lambda x: -10 * math.log10(x)
for tr in ["I", "H16"]:
    print(f"\n### {tr}\n")
    print("| Linear | GEMM | FLIP 비율 A / B | A operand NMSE ×1e-3: E2M1 → +FLIP | GEMM 출력 NMSE ×1e-3: E2M1 → +FLIP | GEMM 출력 SQNR dB |")
    print("|---|---|---|---|---|---|")
    tb = {"A": [0, 0], "B": [0, 0]}; acc = []
    for mo, (O, K) in shape.items():
        sz = {"fprop": (N * K, O * K), "dgrad": (N * O, K * O), "wgrad": (O * N, K * N)}
        for g in ["fprop", "dgrad", "wgrad"]:
            a, b = d[(mo, g, tr, "E2M1")], d[(mo, g, tr, "E2M1|FLIP")]
            for k, sh, n in [("A", b["share_A"][1], sz[g][0] / 16), ("B", b["share_B"][1], sz[g][1] / 16)]:
                tb[k][0] += n; tb[k][1] += sh * n
            acc.append((g, a["g_nmse"], b["g_nmse"], a["A_nmse"], b["A_nmse"]))
            print(f"| {mo} | {g} | {100*b['share_A'][1]:.1f}% / {100*b['share_B'][1]:.1f}% | {a['A_nmse']*1e3:.2f} → {b['A_nmse']*1e3:.2f} ({100*(b['A_nmse']/a['A_nmse']-1):+.1f}%) "
                  f"| {a['g_nmse']*1e3:.2f} → {b['g_nmse']*1e3:.2f} ({100*(b['g_nmse']/a['g_nmse']-1):+.1f}%) | {db(a['g_nmse']):.2f} → {db(b['g_nmse']):.2f} |")
    print(f"\nFLIP block share (block-count weighted): A {100*tb['A'][1]/tb['A'][0]:.1f}%, B {100*tb['B'][1]/tb['B'][0]:.1f}%, all {100*(tb['A'][1]+tb['B'][1])/(tb['A'][0]+tb['B'][0]):.1f}%")
    for g in ["fprop", "dgrad", "wgrad", None]:
        xs = [x for x in acc if g is None or x[0] == g]
        m = lambda i: sum(x[i] for x in xs) / len(xs)
        print(f"- {g or 'all'}: GEMM NMSE {m(1)*1e3:.2f} → {m(2)*1e3:.2f} ({100*(m(2)/m(1)-1):+.1f}%), A operand {m(3)*1e3:.2f} → {m(4)*1e3:.2f} ({100*(m(4)/m(3)-1):+.1f}%)")
