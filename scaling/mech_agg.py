"""Aggregate scale_mech.py -> hf/mech_tables.md (means over decoder Linears; dB gains relative to I with the same rule)."""
import json, math
from pathlib import Path
D = Path(__file__).resolve().parents[1]
db = lambda x, y: 10 * math.log10(x / y)
out = ["# Why a diagonal scale: decomposition of the FPROP GEMM error (`scale_mech.py`)", "",
       "gain = 10 log10(X_I / X_T) per layer (positive = T helps), mean over decoder Linears; gemm = measured, S/nu/c = factors of",
       "pred = S * nu * c. side_A = X-side share of pred; X-side split over top-1% outlier channels / their block-mates / other",
       "blocks, and over dead-zone (6|x|/max < 0.25) / [4,6) elements.", ""]
for tag in ("qwen2.5-0.5b", "olmo2-1b", "qwen3-1.7b"):
    p = D / f"hf/mech_{tag}.jsonl"
    if not p.exists(): continue
    R = [json.loads(l) for l in open(p)]
    out += [f"## {tag} ({len(R)} Linears)", "",
            "| T, rule | gemm gain | S gain | nu gain | c gain | side_A | A: outlier / mates / rest | A: dead-zone / [4,6) |",
            "|---|---|---|---|---|---|---|---|"]
    for rule in ("@6", "@4/6"):
        for T in ("I", "H16", "SM", "QDS"):
            k, k0 = f"{T}{rule}", f"I{rule}"
            m = lambda f: sum(f(r) for r in R) / len(R)
            g = {x: m(lambda r: db(r[k0][x], r[k][x])) for x in ("gemm", "S", "nu", "c")}
            g0 = m(lambda r: db(r["I@6"]["gemm"], r[k]["gemm"]))
            sp = lambda x: m(lambda r: r[k].get(x, float("nan")))
            split = f"{100*sp('A_out'):.0f}% / {100*sp('A_mate'):.0f}% / {100*sp('A_rest'):.0f}%" if T != "H16" else "-"
            out.append(f"| {T} {rule} | {g['gemm']:+.2f} (vs I@6 {g0:+.2f}) | {g['S']:+.2f} | {g['nu']:+.2f} | {g['c']:+.2f} | "
                       f"{100*sp('side_A'):.0f}% | {split} | {100*sp('A_dz'):.0f}% / {100*sp('A_top'):.0f}% |")
    out.append("")
(D / "hf/mech_tables.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
